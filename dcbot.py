import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import asyncio
import random
from typing import Optional
import asyncpg
import signal

# ============================
#   LÄS MILJÖVARIABLER
# ============================
load_dotenv()
token = os.getenv("DISCORD_TOKEN")
GEMINI_KEY = os.getenv("GEMMA_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

print("DEBUG - GROQ_API_KEY =", "SATT" if groq_key else "SAKNAS")
if not token:
    print("VARNING: DISCORD_TOKEN saknas i miljövariablerna!")
if not groq_key:
    print("VARNING: GROQ_API_KEY saknas i miljövariablerna!")

# ============================
#   LOGGNING
# ============================
handler = logging.FileHandler(filename='dcbot.log', encoding='utf-8', mode='a')

OWNER_ID = 117317459819757575
ADMIN_IDS = {112253106359664640}
TIMEZONE = ZoneInfo('Europe/Stockholm')

# ============================
#   DATABASE (Railway Postgres)
# ============================
db_pool: Optional[asyncpg.Pool] = None
db_ready = False

def require_db() -> asyncpg.Pool:
    """Returns pool or raises a clear error if DB isn't ready yet."""
    if db_pool is None:
        raise RuntimeError("DB not ready (db_pool is None)")
    return db_pool


def db_is_ready() -> bool:
    return db_ready and db_pool is not None


async def init_db():
    global db_pool
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL saknas. Lägg till PostgreSQL i Railway.")

    db_pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=5)

    pool = require_db()
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS economy (
            user_id BIGINT PRIMARY KEY,
            balance BIGINT NOT NULL DEFAULT 0,
            last_daily TIMESTAMPTZ
        );
        """)

        await conn.execute(
            "ALTER TABLE economy ADD COLUMN IF NOT EXISTS has_bet BOOLEAN NOT NULL DEFAULT FALSE;"
        )
        await conn.execute(
            "ALTER TABLE economy ADD COLUMN IF NOT EXISTS welcome_claimed BOOLEAN NOT NULL DEFAULT FALSE;"
        )

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS presence_intervals (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            start_ts TIMESTAMPTZ NOT NULL,
            end_ts   TIMESTAMPTZ
        );
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_presence_user_end_start
            ON presence_intervals (user_id, end_ts, start_ts);
        """)
        
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            game TEXT NOT NULL,                 -- "roulette", "coinflip", "blackjack"
            stake BIGINT NOT NULL,              -- how much user bet
            payout BIGINT NOT NULL,             -- how much returned to user (0 if loss)
            profit BIGINT NOT NULL,             -- payout - stake
            result_text TEXT,                   -- optional text (e.g. "heads", "dealer_bust")
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)

        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_user_time
        ON bets (user_id, created_at DESC);
        """)



# ============================
#   DB HELPERS: ECONOMY
# ============================
async def econ_get(user_id: int) -> tuple[int, Optional[datetime]]:
    pool = require_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO economy(user_id)
            VALUES($1)
            ON CONFLICT (user_id) DO NOTHING
            RETURNING balance, last_daily
        """, user_id)

        if row is not None:
            return int(row["balance"]), row["last_daily"]

        row2 = await conn.fetchrow(
            "SELECT balance, last_daily FROM economy WHERE user_id=$1",
            user_id
        )
        return int(row2["balance"]), row2["last_daily"]


async def econ_try_withdraw(user_id: int, amount: int) -> Optional[int]:
    pool = require_db()
    async with pool.acquire() as conn:
        # ensure row exists (no race)
        await conn.execute("""
            INSERT INTO economy(user_id)
            VALUES($1)
            ON CONFLICT (user_id) DO NOTHING
        """, user_id)

        row = await conn.fetchrow("""
            UPDATE economy
            SET balance = balance - $2
            WHERE user_id = $1 AND balance >= $2
            RETURNING balance
        """, user_id, amount)

        if row is None:
            return None
        return int(row["balance"])



async def econ_deposit(user_id: int, amount: int) -> int:
    pool = require_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO economy(user_id, balance)
            VALUES($1, $2)
            ON CONFLICT (user_id) DO UPDATE
            SET balance = economy.balance + EXCLUDED.balance
            RETURNING balance
        """, user_id, amount)
        return int(row["balance"])



async def econ_transfer(sender_id: int, receiver_id: int, amount: int):
    if amount <= 0:
        return None

    pool = require_db()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # ensure both exist
            await conn.execute("""
                INSERT INTO economy(user_id)
                VALUES($1)
                ON CONFLICT (user_id) DO NOTHING
            """, sender_id)
            await conn.execute("""
                INSERT INTO economy(user_id)
                VALUES($1)
                ON CONFLICT (user_id) DO NOTHING
            """, receiver_id)

            # Lock in consistent order to avoid deadlocks
            a, b = sorted([sender_id, receiver_id])
            await conn.execute(
                "SELECT 1 FROM economy WHERE user_id=$1 FOR UPDATE",
                a
            )
            await conn.execute(
                "SELECT 1 FROM economy WHERE user_id=$1 FOR UPDATE",
                b
            )

            # withdraw from sender (must have enough)
            row = await conn.fetchrow("""
                UPDATE economy
                SET balance = balance - $2
                WHERE user_id = $1 AND balance >= $2
                RETURNING balance
            """, sender_id, amount)

            if row is None:
                return None

            # deposit to receiver
            row2 = await conn.fetchrow("""
                UPDATE economy
                SET balance = balance + $2
                WHERE user_id = $1
                RETURNING balance
            """, receiver_id, amount)

            new_sender = int(row["balance"])
            new_receiver = int(row2["balance"])
            return new_sender, new_receiver





WELCOME_BONUS = 300

async def econ_claim_welcome(user_id: int, bonus: int = WELCOME_BONUS) -> Optional[int]:
    pool = require_db()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1) Ensure row exists
            await conn.execute("""
                INSERT INTO economy(user_id)
                VALUES($1)
                ON CONFLICT (user_id) DO NOTHING
            """, user_id)

            # 2) Lock row + read flags
            row = await conn.fetchrow("""
                SELECT balance, has_bet, welcome_claimed
                FROM economy
                WHERE user_id=$1
                FOR UPDATE
            """, user_id)

            # (Row should always exist now, but safety anyway)
            if row is None:
                return None

            if bool(row["has_bet"]) or bool(row["welcome_claimed"]):
                return None

            # 3) Apply bonus + mark claimed, return new balance
            updated = await conn.fetchrow("""
                UPDATE economy
                SET balance = balance + $2,
                    welcome_claimed = TRUE
                WHERE user_id = $1
                RETURNING balance
            """, user_id, bonus)

            return int(updated["balance"]) if updated else None



# ============================
#   DB HELPERS: PRESENCE
# ============================
async def presence_get_last_7_days(user_id: int) -> list[tuple[datetime, datetime]]:
    pool = require_db()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT start_ts, end_ts
            FROM presence_intervals
            WHERE user_id=$1
              AND end_ts IS NOT NULL
              AND end_ts >= NOW() - INTERVAL '7 days'
            ORDER BY start_ts ASC
        """, user_id)

    return [(r["start_ts"], r["end_ts"]) for r in rows]



async def presence_close_all_open_sessions(user_id: int, end_dt: datetime):
    if db_pool is None:
        return  # silent best-effort
    pool = require_db()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE presence_intervals
            SET end_ts = $2
            WHERE user_id = $1 AND end_ts IS NULL
        """, user_id, end_dt)




last_status: Optional[discord.Status] = None
last_change_time: datetime | None = None




def is_online_like(status: discord.Status) -> bool:
    return status in (
        discord.Status.online,
        discord.Status.idle,
        discord.Status.dnd,
    )


async def presence_open_session(user_id: int, start_dt: datetime):
    pool = require_db()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO presence_intervals(user_id, start_ts, end_ts)
            VALUES($1, $2, NULL)
        """, user_id, start_dt)


async def presence_close_session(user_id: int, end_dt: datetime):
    pool = require_db()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE presence_intervals
            SET end_ts = $2
            WHERE id = (
                SELECT id
                FROM presence_intervals
                WHERE user_id = $1 AND end_ts IS NULL
                ORDER BY start_ts DESC
                LIMIT 1
            )
        """, user_id, end_dt)


async def presence_has_open_session(user_id: int) -> bool:
    pool = require_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT 1
            FROM presence_intervals
            WHERE user_id=$1 AND end_ts IS NULL
            LIMIT 1
        """, user_id)
    return row is not None

async def log_bet(user_id: int, game: str, stake: int, payout: int, profit: int, result_text: str = ""):
    pool = require_db()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO bets(user_id, game, stake, payout, profit, result_text)
            VALUES($1, $2, $3, $4, $5, $6)
        """, user_id, game, stake, payout, profit, result_text)


async def stats_for_user(user_id: int):
    pool = require_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END), 0) AS wins,
                COALESCE(SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END), 0) AS losses,
                COALESCE(MAX(profit), 0) AS biggest_win,
                COALESCE(MIN(profit), 0) AS biggest_loss,
                COALESCE(SUM(profit), 0) AS net_profit
            FROM bets
            WHERE user_id=$1
        """, user_id)
    return row


async def global_biggest_win_loss():
    pool = require_db()
    async with pool.acquire() as conn:
        biggest_win = await conn.fetchrow("""
            SELECT user_id, game, stake, payout, profit, result_text, created_at
            FROM bets
            ORDER BY profit DESC
            LIMIT 1
        """)
        biggest_loss = await conn.fetchrow("""
            SELECT user_id, game, stake, payout, profit, result_text, created_at
            FROM bets
            ORDER BY profit ASC
            LIMIT 1
        """)
    return biggest_win, biggest_loss


async def last_bets_for_user(user_id: int, limit: int = 10):
    pool = require_db()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT game, stake, profit, result_text, created_at
            FROM bets
            WHERE user_id=$1
            ORDER BY created_at DESC
            LIMIT $2
        """, user_id, limit)
    return rows


# ============================
#   DISCORD INTENTS & BOT
# ============================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True

class MyBot(commands.Bot):
    async def close(self):
        # run your cleanup once, then let discord.py close
        await shutdown()
        await super().close()

bot = MyBot(command_prefix='!', intents=intents)


async def shutdown():
    global db_pool, db_ready
    print("🛑 Shutdown started...")

    try:
        now = datetime.now(TIMEZONE)
        await presence_close_all_open_sessions(OWNER_ID, now)
        print("✅ Closed open presence sessions (if any)")
    except Exception as e:
        print("⚠️ Failed to close presence sessions:", repr(e))

    if db_pool is not None:
        await db_pool.close()
        db_pool = None
        db_ready = False
        print("🛑 DB pool closed")




@bot.event
async def setup_hook():
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(_shutdown_from_loop()))
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(_shutdown_from_loop()))
        print("✅ Signal handlers registered")
    except (NotImplementedError, AttributeError):
        print("⚠️ Signal handlers not supported on this platform.")



async def _shutdown_from_loop():
    await bot.close()



@bot.event
async def on_disconnect():
    print("⚠️ Discord disconnected (will try to reconnect)")




# ============================
#   KORTTIDSMINNE FÖR GPT
# ============================
# Nyckel: channel_id (eller user_id om du vill per person)
# Värde: lista av {"role": "...", "content": "..."}
conversation_history: dict[int, list[dict]] = {}

@bot.event
async def on_ready():
    global db_ready, last_status, last_change_time

    if not db_ready:
        try:
            await init_db()
            db_ready = True
            print("✅ DB connected")
        except Exception as e:
            print("❌ DB init failed:", repr(e))
            return

    print(f'Logged in as {bot.user} (ID: {bot.user.id})')

    owner_member = None
    for guild in bot.guilds:
        m = guild.get_member(OWNER_ID)
        if m is not None:
            owner_member = m
            break

    now = datetime.now(TIMEZONE)

    if owner_member is None:
        print("VARNING: Kunde inte hitta OWNER_ID i någon guild.")
        last_status = discord.Status.offline
        last_change_time = now
        return

    current_status = owner_member.status
    last_status = current_status

    if is_online_like(current_status):
        last_change_time = now
        if not await presence_has_open_session(OWNER_ID):
            await presence_open_session(OWNER_ID, now)
    else:
        last_change_time = None
        if await presence_has_open_session(OWNER_ID):
            await presence_close_session(OWNER_ID, now)

    print(f"Startstatus: {last_status}, last_change_time: {last_change_time}")



#   SIX SEVEN RESPONSE
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if '67' in message.content.lower():
        await message.channel.send('SIX SEVEN!')

    await bot.process_commands(message)

@bot.event
async def on_presence_update(before: discord.Member, after: discord.Member):
    global last_status, last_change_time

    if before.id != OWNER_ID:
        return

    now = datetime.now(TIMEZONE)

    if last_status is None:
        last_status = before.status
        last_change_time = now
        return

    was_online = is_online_like(last_status)
    is_now_online = is_online_like(after.status)

    # OFFLINE: close open session
    if was_online and not is_now_online:
        if await presence_has_open_session(OWNER_ID):
            await presence_close_session(OWNER_ID, now)
        last_change_time = None

    # ONLINE: open new session
    if not was_online and is_now_online:
        last_change_time = now
        if not await presence_has_open_session(OWNER_ID):
            await presence_open_session(OWNER_ID, now)

    last_status = after.status


@bot.command(help="Säger hej till dig. Mest för debug")
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}!!')

@bot.command(help="Skickar Caramelldansen-lyrics.")
async def dansa(ctx):
    await ctx.send('Du-du-du Hey-yeah-yeah-i-yeah \n' 'Vi undrar, är ni redo att vara med? \n' 'Armarna upp, nu ska ni få se \n' 'Kom igen Vem som helst kan vara med (Vara med) \n' 'Så rör på era fötter, o-a-a-a \n' 'Och vicka era höfter, o-la-la-la \n' 'Gör som vi. Till denna melodi \n' 'O-a, o-a-a \n' 'Dansa med oss, klappa era händer \n' 'Gör som vi gör, ta några steg åt vänster \n' 'Lyssna och lär, missa inte chansen \n' 'Nu är vi här med caramelldansen')

@bot.command(help="Visar exakta onlinetider för hebbe senaste 7 dagarna.")
async def hebbe(ctx):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return
    """Visar exakta onlinetider (start–slut) per dag senaste 7 dagarna."""
    now = datetime.now(TIMEZONE)
    seven_days_ago = now.date() - timedelta(days=6)

    weekday_names = ['Måndag', 'Tisdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lördag', 'Söndag']

    # dag -> lista av (start, slut)
    per_day: dict[datetime.date, list[tuple[datetime, datetime]]] = {}
    interval_rows = await presence_get_last_7_days(OWNER_ID)

    for start, end in interval_rows:
        start = start.astimezone(TIMEZONE)
        end = end.astimezone(TIMEZONE)


        # hoppa över intervall helt utanför 7-dagarsfönstret
        if end.date() < seven_days_ago or start.date() > now.date():
            continue

        # klipp intervallen till [seven_days_ago, now]
        if start.date() < seven_days_ago:
            start = datetime.combine(seven_days_ago, datetime.min.time(), tzinfo=TIMEZONE)
        if end > now:
            end = now

        # dela upp över midnatt
        current = start
        while current < end:
            day = current.date()
            next_midnight = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=TIMEZONE)
            segment_end = min(end, next_midnight)

            per_day.setdefault(day, []).append((current, segment_end))
            current = segment_end

    def merge_intervals(intervals: list[tuple[datetime, datetime]]):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    lines = []

    for i in range(7):
        day = seven_days_ago + timedelta(days=i)
        weekday = weekday_names[day.weekday()]

        intervals = per_day.get(day, [])
        intervals = merge_intervals(intervals)

        if not intervals:
            lines.append(f'{weekday}: -')
        else:
            parts = [f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in intervals]
            lines.append(f"{weekday}: " + " & ".join(parts))

    msg = '**Onlinetider (senaste 7 dagarna):**\n```text\n' + '\n'.join(lines) + '\n```'
    await ctx.send(msg)


# ============================
#   GPT / GROQ-KOMMANDO
# ============================
@bot.command(name="gpt", help="Chatta med schizo-AI.")
async def gpt(ctx, *, prompt: str | None = None):
    """Skickar prompten till en sån där AI och återkommer."""
    print(f"!gpt triggat av {ctx.author} med prompt: {prompt!r}")

    if not prompt:
        await ctx.reply("Du måste skriva något efter kommandot idiot, t.ex. `!gpt skriv en dikt om 67`")
        return

    GROQ_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_KEY:
        await ctx.reply("GROQ_API_KEY saknas! Lägg till den i Railway.")
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }

    # ===== HÄR KOMMER MINNET IN =====
    channel_id = ctx.channel.id
    history = conversation_history.get(channel_id, [])

    # Begränsa historiken så den inte blir gigantisk (t.ex. senaste 8 meddelandena)
    max_history_messages = 8
    if len(history) > max_history_messages:
        history = history[-max_history_messages:]

    messages = [
        {
            "role": "system",
            "content": (
                "Du är en snabb och hjälpsam Discord-bot. "
                "Svara kort, roligt och ungt. Släng gärna in lite meme / internet slang."
            ),
        },
        # lägg till historiken
        *history,
        # nya user-meddelandet
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.7,
    }

    print("[Groq] Förbereder request...")

    try:
        async with ctx.channel.typing():
            print("[Groq] Skickar request...")
            resp = await asyncio.to_thread(
                requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=15
            )

            

        print("[Groq] Statuskod:", resp.status_code)
        print("[Groq] Förhandsvisning av svar:", resp.text[:300])

        if resp.status_code != 200:
            await ctx.reply(f"Groq API-fel {resp.status_code}:\n```{resp.text[:300]}```")
            return

        try:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
        except Exception as e:
            print("[Groq] JSON-fel:", repr(e))
            await ctx.reply("Kunde inte tolka svaret från Groq.")
            return

        if not reply:
            reply = "Jag fick ett tomt svar walla"

        # ===== UPPDATERA MINNET =====
        # Lägg till både user & bot-svar i historiken
        new_history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
        # Trimma igen
        if len(new_history) > max_history_messages:
            new_history = new_history[-max_history_messages:]
        conversation_history[channel_id] = new_history

        # Skicka svaret
        if len(reply) <= 2000:
            await ctx.reply(reply)
        else:
            for i in range(0, len(reply), 1900):
                await ctx.send(reply[i:i+1900])

    except Exception as e:
        print("[Groq] Oväntat fel i gpt-kommandot:", repr(e))
        await ctx.reply(f"Något gick snett i gpt-kommandot:\n`{repr(e)}`")



@bot.command(help="Nollställer AI-minnet, om den fuckar ur.")
async def resetgpt(ctx):
    """Nollställer korttidsminnet i den här kanalen."""
    channel_id = ctx.channel.id
    if channel_id in conversation_history:
        del conversation_history[channel_id]
        await ctx.reply("Jag har glömt allt vi snackat om i den här kanalen 😵‍💫")
    else:
        await ctx.reply("Jag hade inget minne sparat här ändå.")



# ============================
#   GEMMA / GOOGLE AI KOMMANDO
# ============================
@bot.command(name="gg", help="Fråga en bättre AI, ger vettiga svar.")
async def gg(ctx, *, prompt: str | None = None):
    """Fråga Google Gemini (flash-modellen)."""
    print(f"!gg triggat av {ctx.author} med prompt: {prompt!r}")

    if not prompt:
        await ctx.reply("Skriv något efter kommandot, t.ex. `!gg vad är 67 + 67?`")
        return

    GEMINI_KEY = os.getenv("GEMMA_API_KEY")  # eller byt till GEMINI_API_KEY om du vill
    if not GEMINI_KEY:
        await ctx.reply("GEMMA_API_KEY saknas! Lägg till den i Railway (Google AI Studio-nyckeln).")
        return

    # OBS: nu kör vi Gemini 2.5 Flash – detta är ett giltigt model-ID för v1beta
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
        f"?key={GEMINI_KEY}"
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    print("[Gemini] Skickar request...")

    try:
        async with ctx.channel.typing():
            resp = await asyncio.to_thread(
                requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=15
            )


        print("[Gemini] Statuskod:", resp.status_code)
        print("[Gemini] Preview:", resp.text[:300])

        if resp.status_code != 200:
            await ctx.reply(f"Gemini API-fel {resp.status_code}:\n```{resp.text[:300]}```")
            return

        data = resp.json()

        # "candidates" -> "content" -> "parts"[0]["text"]
        try:
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print("[Gemini] JSON-fel:", repr(e))
            await ctx.reply("Kunde inte tolka svaret från Gemini.")
            return

        if not reply:
            reply = "Gemini gav inget svar, wtf ¯\\_(ツ)_/¯"

        if len(reply) <= 2000:
            await ctx.reply(reply)
        else:
            for i in range(0, len(reply), 1900):
                await ctx.send(reply[i:i+1900])

    except Exception as e:
        print("[Gemini] Oväntat fel:", repr(e))
        await ctx.reply(f"Något gick snett i !gg-kommandot:\n`{repr(e)}`")


@bot.command(name="summary1h", help="Sammanfattar chatten den senaste timmen.")
async def summary1h(ctx):
    """Sammanfattar alla meddelanden i den här kanalen senaste timmen med Gemini."""
    print(f"[summary1h] triggat av {ctx.author} i kanal {ctx.channel.id}")

    try:
        now = datetime.now(TIMEZONE)
        since = now - timedelta(hours=1)

        await ctx.reply("(debug) Hämtar meddelanden senaste timmen...")

        messages: list[str] = []

        print("[summary1h] startar history-fetch...")
        async for msg in ctx.channel.history(limit=400, after=since, oldest_first=True):
            # debug så vi ser att loopen faktiskt körs
            print(f"[summary1h] hittade meddelande från {msg.author}: {msg.content[:50]!r}")

            # Skippa bots
            if msg.author.bot:
                continue

            content = msg.content.strip()
            if not content:
                continue

            messages.append(f"{msg.author.display_name}: {content}")

        print(f"[summary1h] antal user-meddelanden: {len(messages)}")

        if not messages:
            await ctx.send("Det finns inga meddelanden senaste timmen att sammanfatta.")
            return

        # Bygg transcript
        transcript = "\n".join(messages)

        # Klipp om det är superlångt
        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[-max_chars:]
            print("[summary1h] transcript nerklippt till", max_chars, "tecken")

        # Prompt till Gemini – samma modell som !gg
        summary_prompt = (
            "Du är en AI-assistent som sammanfattar en Discord-kanal.\n"
            "Här är meddelanden från den senaste timmen (äldre först):\n\n"
            f"{transcript}\n\n"
            "Uppgift:\n"
            "- Skriv en tydlig sammanfattning på svenska (ca 5–10 meningar).\n"
            "- Ta upp viktiga ämnen, beslut, frågor och skämt.\n"
            "- Hitta inte på saker som inte nämns i texten.\n"
        )

        GEMINI_KEY = os.getenv("GEMMA_API_KEY")
        if not GEMINI_KEY:
            await ctx.send("GEMMA_API_KEY saknas! Lägg till den i Railway (Google AI Studio-nyckeln).")
            return

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:generateContent"
            f"?key={GEMINI_KEY}"
        )

        payload = {
            "contents": [
                {"parts": [{"text": summary_prompt}]}
            ]
        }

        headers = {"Content-Type": "application/json"}

        print("[Gemini-summary] Skickar request...")
        async with ctx.channel.typing():
            resp = await asyncio.to_thread(
                requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=20
            )


        print("[Gemini-summary] Statuskod:", resp.status_code)
        print("[Gemini-summary] Preview:", resp.text[:300])

        if resp.status_code != 200:
            await ctx.send(f"Gemini API-fel {resp.status_code}:\n```{resp.text[:300]}```")
            return

        try:
            data = resp.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print("[Gemini-summary] JSON-fel:", repr(e), "BODY:", resp.text[:500])
            await ctx.send("Kunde inte tolka sammanfattningssvaret från Gemini.")
            return

        if not reply:
            reply = "Gemini gav ingen sammanfattning, wtf ¯\\_(ツ)_/¯"

        # Skicka sammanfattningen (splitta om för lång)
        if len(reply) <= 2000:
            await ctx.send(reply)
        else:
            for i in range(0, len(reply), 1900):
                await ctx.send(reply[i:i+1900])

    except Exception as e:
        import traceback
        traceback.print_exc()
        await ctx.send(f"summary1h kraschade:\n`{type(e).__name__}: {e}`")

#Till ECON FÖR GAMBA
DAILY_COINS = 100
MAX_BET = DAILY_COINS * 10
rng = random.SystemRandom()


# Roulette helpers (European roulette: 0-36, 0 is green)
RED_NUMS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK_NUMS = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}

def roulette_color(n: int) -> str:
    if n == 0:
        return "green"
    return "red" if n in RED_NUMS else "black"

def color_emoji(color: str) -> str:
    return {"red": "🔴", "black": "⚫", "green": "🟢"}[color]

# European wheel order (single-zero)
WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5,
    24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

def tile(n: int) -> str:
    col = roulette_color(n)
    # compact fixed-width tile for nice alignment in code blocks
    return f"{color_emoji(col)}{n:>2}"

def render_strip(start_idx: int, width: int = 10) -> str:
    parts = []
    for i in range(width):
        n = WHEEL[(start_idx + i) % len(WHEEL)]
        parts.append(tile(n))

    # pointer in the middle
    pointer_pos = width // 2
    parts[pointer_pos] = f"⬇️{parts[pointer_pos]}⬇️"
    return " | ".join(parts)






BJ_ACTIVE: set[tuple[int, int]] = set()  # (channel_id, user_id)

def bj_value(cards: list[str]) -> int:
    # cards are like "A♠", "10♥", "K♦"
    ranks = [c[:-1] for c in cards]
    total = 0
    aces = 0
    for r in ranks:
        if r in ("J","Q","K"):
            total += 10
        elif r == "A":
            aces += 1
            total += 11
        else:
            total += int(r)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total

def draw_card(deck: list[str]) -> str:
    return deck.pop()

def new_deck() -> list[str]:
    suits = ["♠","♥","♦","♣"]
    ranks = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    deck = [r+s for s in suits for r in ranks]
    rng.shuffle(deck)
    return deck




@bot.command(help="Få 100 coins var 24:e timme.")
async def daily(ctx):
    if not db_is_ready():
        await ctx.reply("Databasen startar fortfarande, testa igen om några sekunder.")
        return

    now = datetime.now(TIMEZONE)
    pool = require_db()

    async with pool.acquire() as conn:
        async with conn.transaction():
            # ensure row exists
            await conn.execute("""
                INSERT INTO economy(user_id)
                VALUES($1)
                ON CONFLICT (user_id) DO NOTHING
            """, ctx.author.id)

            row = await conn.fetchrow("""
                SELECT balance, last_daily
                FROM economy
                WHERE user_id=$1
                FOR UPDATE
            """, ctx.author.id)

            balance = int(row["balance"])
            last_daily = row["last_daily"]

            if last_daily is not None:
                last_local = last_daily.astimezone(TIMEZONE)
                if now - last_local < timedelta(hours=24):
                    remaining = timedelta(hours=24) - (now - last_local)
                    hours, rem = divmod(int(remaining.total_seconds()), 3600)
                    mins = rem // 60
                    await ctx.reply(f"Du har redan claimat daily. Kom tillbaka om **{hours}h {mins}m**.")
                    return

            await conn.execute(
                "UPDATE economy SET balance = balance + $1, last_daily = $2 WHERE user_id = $3",
                DAILY_COINS, now, ctx.author.id
            )

            # fetch new balance (optional but nice)
            row2 = await conn.fetchrow("SELECT balance FROM economy WHERE user_id=$1", ctx.author.id)
            balance = int(row2["balance"])

    await ctx.reply(f"✅ {ctx.author.mention} fick **{DAILY_COINS} coins**! Ny balans: **{balance}** 💰")




# BALANCE COMMAND
@bot.command(aliases=["bal"], help="Visar din coin-balante.")
async def balance(ctx, member: discord.Member | None = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return
    member = member or ctx.author
    bal, _ = await econ_get(member.id)
    await ctx.reply(f"💰 {member.display_name} har **{bal} cöins**.")



# LEADERBOARD COMMAND
@bot.command(help="Visar top 10 rikaste spelarna i servern.")
async def leaderboard(ctx):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    pool = require_db()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT user_id, balance
            FROM economy
            ORDER BY balance DESC
            LIMIT 10
        """)

    if not rows:
        await ctx.reply("Ingen har coins än. Skriv `!daily` för att starta economy.")
        return

    lines = []
    rank = 1
    for r in rows:
        member = ctx.guild.get_member(int(r["user_id"]))
        if member is None:
            continue
        lines.append(f"**#{rank}** {member.display_name} — **{int(r['balance'])}** 💰")
        rank += 1

    embed = discord.Embed(title="🏆 Leaderboard (Top 10)", description="Mest cash på servern", color=0xF1C40F)
    embed.add_field(name="Ranking", value="\n".join(lines) if lines else "—", inline=False)
    embed.set_footer(text=f"Max bet: {MAX_BET} | Daily: {DAILY_COINS}/24h")
    await ctx.send(embed=embed)



#BET COMMAND
#endast red black green
#max bet 1000
#cooldown är 1bet / 3 sec
@bot.command(help="Bettar coins på red, black eller green. Ex: !bet 100 red eller !bet all green")
@commands.cooldown(1, 3, commands.BucketType.user)
async def bet(ctx, amount: str = None, color: str = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return
    if amount is None or color is None:
        await ctx.reply("Ex: `!bet 100 red` | `!bet all black` | `!bet 50 green`")
        return

    color = color.lower().strip()
    if color not in ("red", "black", "green"):
        await ctx.reply("Du kan bara betta på: `red`, `black`, `green`.")
        return

    bal, _ = await econ_get(ctx.author.id)

    if amount.lower() == "all":
        if bal <= 0:
            await ctx.reply("Du har **0 coins**. Claim `!daily` först kanske")
            return
        bet_amount = min(bal, MAX_BET)
        all_in = True
    else:
        try:
            bet_amount = int(amount)
        except:
            await ctx.reply("Amount måste vara ett tal eller `all`.")
            return

        if bet_amount <= 0:
            await ctx.reply("Amount måste vara > 0.")
            return
        if bet_amount > MAX_BET:
            await ctx.reply(f"Max bet är **{MAX_BET}** (10x daily).")
            return
        all_in = False

    # Withdraw stake atomically
    new_bal_after_withdraw = await econ_try_withdraw(ctx.author.id, bet_amount)
    if new_bal_after_withdraw is None:
        await ctx.reply(f"Du har inte tillräckligt med coins brokeboy (balans: **{bal}**).")
        return

    # Mark that user has placed a bet (ever)
    pool = require_db()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE economy SET has_bet=TRUE WHERE user_id=$1",
            ctx.author.id
        )

    # -----------------------------
    # Spin animation (5-step fast spin)
    # -----------------------------
    result_num = rng.randrange(0, 37)
    result_col = roulette_color(result_num)
    target_idx = WHEEL.index(result_num)
    
    strip_width = 10
    pointer_pos = strip_width // 2
    landing_start = (target_idx - pointer_pos) % len(WHEEL)


    spin_msg = await ctx.send(
        f"🎰 {ctx.author.mention} bettar **{bet_amount}** på {color_emoji(color)} **{color}**...\n"
        f"```text\n{render_strip(0)}\n```"
    )

    async def safe_edit(content: str) -> bool:
        try:
            await spin_msg.edit(content=content)
            return True
        except discord.NotFound:
            return False

    idx = rng.randrange(0, len(WHEEL))

    # EDIT 1 – instant start
    idx = (idx + rng.randrange(6, 10)) % len(WHEEL)
    await safe_edit(f"🎰 Snurrar...\n```text\n{render_strip(idx, width=strip_width)}\n```")
    await asyncio.sleep(0.12)

    # EDIT 2 – fast jump
    idx = (idx + rng.randrange(6, 10)) % len(WHEEL)
    await safe_edit(f"🎰 Snurrar...\n```text\n{render_strip(idx, width=strip_width)}\n```")
    await asyncio.sleep(0.12)

    # EDIT 3 – slower approach
    dist = (landing_start - idx) % len(WHEEL)
    idx = (idx + max(1, dist // 2)) % len(WHEEL)
    await safe_edit(f"🎰 Snurrar...\n```text\n{render_strip(idx, width=strip_width)}\n```")
    await asyncio.sleep(0.15)

    # EDIT 4 – almost there
    dist = (landing_start - idx) % len(WHEEL)
    idx = (idx + max(1, dist - 1)) % len(WHEEL)
    await safe_edit(f"🎰 Snurrar...\n```text\n{render_strip(idx, width=strip_width)}\n```")
    await asyncio.sleep(0.18)

    # EDIT 5 – final landing (exact result)
    final_strip = f"```text\n{render_strip(landing_start, width=strip_width)}\n```"
    await safe_edit(f"🎰 Snurrar...\n{final_strip}")
    await asyncio.sleep(0.10)




    # -----------------------------
    # Resolve bet + payout
    # -----------------------------
    win = (result_col == color)

    if color in ("red", "black"):
        win_total_return = bet_amount * 2
        profit = bet_amount
    else:
        win_total_return = bet_amount * 36
        profit = bet_amount * 35

    if win:
        new_bal = await econ_deposit(ctx.author.id, win_total_return)
        await log_bet(ctx.author.id, "roulette", bet_amount, win_total_return, profit, f"{result_num}/{result_col}/{color}")
        extra = " (ALL-IN)" if all_in else ""
        await spin_msg.edit(content=(
            f"🎯 RESULTAT: {color_emoji(result_col)} **{result_num}**\n"
            f"{final_strip}\n"
            f"✅ {ctx.author.mention} vann{extra}! +**{profit}** profit\n"
            f"💰 Ny balans: **{new_bal}**"
        ))
    else:
        bal_now, _ = await econ_get(ctx.author.id)
        await log_bet(ctx.author.id, "roulette", bet_amount, 0, -bet_amount, f"{result_num}/{result_col}/{color}")
        await spin_msg.edit(content=(
            f"🎯 RESULTAT: {color_emoji(result_col)} **{result_num}**\n"
            f"{final_strip}\n"
            f"❌ {ctx.author.mention} förlorade **{bet_amount}** coins unluko\n"
            f"💰 Ny balante: **{bal_now}**"
        ))



#BET COOLDOWN
@bet.error
async def bet_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.reply(f"Chilla, vänta **{error.retry_after:.1f}s** innan nästa bet.")
    else:
        raise error


#LISTA ALLA COMMANDS
@bot.command(name="commands", help="Visar alla commands.")
async def commands_list(ctx):
    embed = discord.Embed(title="📜 Commands", description="Här är allt jag kan göra:", color=0x5865F2)

    items = []
    for cmd in bot.commands:
        if cmd.hidden:
            continue
        brief = cmd.help or "—"
        items.append((cmd.name, brief))

    items.sort(key=lambda x: x[0])

    # Show a few per field to avoid embed limits
    lines = [f"`!{name}` — {brief}" for name, brief in items]
    chunk = "\n".join(lines[:25])  # keep it safe

    embed.add_field(name="Lista", value=chunk, inline=False)
    embed.set_footer(text="Tips: skriv !daily, !bet, !leaderboard")

    await ctx.send(embed=embed)

@bot.command(help="Skicka cöins till någon. Ex: !send 200 @user")
async def send(ctx, amount: str = None, *, member: discord.Member = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    if amount is None or member is None:
        await ctx.reply("Ex: `!send 200 @user`")
        return

    if member.bot:
        await ctx.reply("Du kan inte skicka coins till en bot.")
        return

    if member.id == ctx.author.id:
        await ctx.reply("Du kan inte skicka coins till dig själv bror")
        return

    # Parse amount
    if amount.lower() == "all":
        bal, _ = await econ_get(ctx.author.id)
        if bal <= 0:
            await ctx.reply("Du har inga coins att skicka idiot.")
            return
        send_amount = bal
    else:
        try:
            send_amount = int(amount)
        except ValueError:
            await ctx.reply("Amount måste vara ett heltal (eller `all`).")
            return

        if send_amount <= 0:
            await ctx.reply("Amount måste vara > 0.")
            return


    # Do atomic transfer
    result = await econ_transfer(ctx.author.id, member.id, send_amount)
    if result is None:
        bal, _ = await econ_get(ctx.author.id)
        await ctx.reply(f"Du har inte tillräckligt, fattiglapp. Din balans: **{bal}** 💰")
        return

    new_sender, new_receiver = result
    await ctx.reply(
        f"✅ {ctx.author.mention} skickade **{send_amount}** coins till {member.mention}\n"
        f"💸 Din balans: **{new_sender}** | {member.display_name}: **{new_receiver}**"
    )

def is_admin():
    async def predicate(ctx: commands.Context):
        return ctx.author.id in ADMIN_IDS
    return commands.check(predicate)

@bot.command(hidden=True, help="(Elliot) Give coins. Ex: !give 100 @user")
@is_admin()
async def give(ctx, amount: str = None, *, member: discord.Member = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    if amount is None or member is None:
        await ctx.reply("Ex: `!give 100 @user`")
        return

    if member.bot:
        await ctx.reply("Du kan inte ge coins till en bot.")
        return

    try:
        amt = int(amount)
    except ValueError:
        await ctx.reply("Amount måste vara ett heltal.")
        return

    if amt <= 0:
        await ctx.reply("Amount måste vara > 0.")
        return

    new_bal = await econ_deposit(member.id, amt)
    await ctx.reply(f"✅ Gav **{amt}** coins till {member.mention}. Ny balans: **{new_bal}** 💰")

@give.error
async def give_error(ctx, error):
    if isinstance(error, commands.CheckFailure):
        await ctx.reply("❌ Nt degen. Du har inte behörighet att använda `!give`.")
        return
    raise error


@bot.command(help="Få en välkomstbonus på 300 coins (bara om du aldrig bettat!!).")
async def welcomebonus(ctx):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    new_bal = await econ_claim_welcome(ctx.author.id, 300)
    if new_bal is None:
        await ctx.reply("❌ Du kan inte claima welcome bonus (antingen redan claimad eller så har du redan bettat).")
        return

    await ctx.reply(f"🎁 {ctx.author.mention} fick **300 coins** i welcome bonus! Ny balans: **{new_bal}** 💰")


@bot.command(help="Stats. !stats | !stats @user | !stats all")
async def stats(ctx, arg: str = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    # !stats all  -> global biggest win/loss
    if arg and arg.lower() == "all":
        win_row, loss_row = await global_biggest_win_loss()
        if not win_row and not loss_row:
            await ctx.reply("Inga bets loggade än.")
            return

        def fmt_row(r, label: str):
            if not r:
                return f"{label}: —"
        
            user_id = int(r["user_id"])
            member = ctx.guild.get_member(user_id)
            name = member.display_name if member else f"User {user_id}"
        
            game = r["game"]
            stake = int(r["stake"])
            profit = int(r["profit"])
            created = r["created_at"].astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M")
        
            # Profit/loss text
            profit_txt = f"profit {profit}" if profit >= 0 else f"loss {abs(profit)}"
        
            # Prettify result_text (supports your old roulette format "num/landedColor/betColor")
            res = (r["result_text"] or "")
            if game == "roulette" and "/" in res:
                parts = res.split("/")
                if len(parts) >= 3:
                    num, landed_col, bet_col = parts[0], parts[1], parts[2]
                    res = f"landed={num}({landed_col}) bet={bet_col}"
        
            return f"{label}: {name} | {game} | stake {stake} | {profit_txt} | {res} | {created}"


        msg = "**📊 Global stats (ALL):**\n" + "```text\n" + fmt_row(win_row, "BIGGEST WIN") + "\n" + fmt_row(loss_row, "BIGGEST LOSS") + "\n```"
        await ctx.reply(msg)
        return

    # !stats @user (or default: self)
    member = None
    if ctx.message.mentions:
        member = ctx.message.mentions[0]
    else:
        member = ctx.author

    row = await stats_for_user(member.id)
    if not row or int(row["total"]) == 0:
        await ctx.reply(f"Inga bets loggade för **{member.display_name}** än.")
        return

    total = int(row["total"])
    wins = int(row["wins"])
    losses = int(row["losses"])
    biggest_win = int(row["biggest_win"])
    biggest_loss = int(row["biggest_loss"])
    net_profit = int(row["net_profit"])

    await ctx.reply(
        f"**📊 Stats för {member.mention}:**\n"
        f"✅ Wins: **{wins}** | ❌ Losses: **{losses}** | 🎲 Total: **{total}**\n"
        f"🏆 Biggest win: **{biggest_win}** | 💀 Biggest loss: **{biggest_loss}**\n"
        f"📈 Net profit: **{net_profit}**"
    )


@bot.command(help="Senaste bets. !bets | !bets @user (admin only)")
async def bets(ctx, member: discord.Member | None = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    member = member or ctx.author

    # Only admins can view others
    if member.id != ctx.author.id and ctx.author.id not in ADMIN_IDS:
        await ctx.reply("❌ Bara admins kan kolla andra personers bet-history.")
        return

    rows = await last_bets_for_user(member.id, limit=10)
    if not rows:
        await ctx.reply(f"Inga bets loggade för **{member.display_name}** än.")
        return

    lines = []
    for r in rows:
        game = r["game"]
        stake = int(r["stake"])
        profit = int(r["profit"])
        when = r["created_at"].astimezone(TIMEZONE).strftime("%m-%d %H:%M")
        res = (r["result_text"] or "")[:24]
        sign = "+" if profit > 0 else ""
        lines.append(f"{when} | {game:<9} | stake {stake:<5} | profit {sign}{profit:<6} | {res}")

    await ctx.reply(f"**🧾 Last 10 bets — {member.display_name}:**\n```text\n" + "\n".join(lines) + "\n```")


@bot.command(help="Coinflip. Ex: !coinflip 100 heads | !coinflip all tails")
@commands.cooldown(1, 3, commands.BucketType.user)
async def coinflip(ctx, amount: str = None, side: str = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    if amount is None or side is None:
        await ctx.reply("Ex: `!coinflip 100 heads` | `!coinflip all tails`")
        return

    side = side.lower().strip()
    if side in ("h", "head"):
        side = "heads"
    if side in ("t", "tail"):
        side = "tails"

    if side not in ("heads", "tails"):
        await ctx.reply("Du måste välja `heads` eller `tails`.")
        return

    bal, _ = await econ_get(ctx.author.id)

    if amount.lower() == "all":
        if bal <= 0:
            await ctx.reply("Du har 0 coins. Claim `!daily` eller tigg av någon.")
            return
        bet_amount = min(bal, MAX_BET)
    else:
        try:
            bet_amount = int(amount)
        except ValueError:
            await ctx.reply("Amount måste vara ett tal eller `all`.")
            return
        if bet_amount <= 0:
            await ctx.reply("Amount måste vara > 0.")
            return
        if bet_amount > MAX_BET:
            await ctx.reply(f"Max bet är **{MAX_BET}**.")
            return

    # take stake
    new_bal_after_withdraw = await econ_try_withdraw(ctx.author.id, bet_amount)
    if new_bal_after_withdraw is None:
        await ctx.reply(f"Du har inte tillräckligt (balans: **{bal}**).")
        return

    flip = rng.choice(["heads", "tails"])
    win = (flip == side)

    if win:
        payout = bet_amount * 2
        profit = bet_amount
        new_bal = await econ_deposit(ctx.author.id, payout)
        await log_bet(ctx.author.id, "coinflip", bet_amount, payout, profit, flip)
        await ctx.reply(f"🪙 Det blev **{flip}**! ✅ Du vann **+{profit}** | Ny balans: **{new_bal}**")
    else:
        await log_bet(ctx.author.id, "coinflip", bet_amount, 0, -bet_amount, flip)
        bal_now, _ = await econ_get(ctx.author.id)
        await ctx.reply(f"🪙 Det blev **{flip}**… ❌ Du förlorade **-{bet_amount}** | Ny balans: **{bal_now}**")


@coinflip.error
async def coinflip_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.reply(f"Chilla, vänta **{error.retry_after:.1f}s**.")
    else:
        raise error



@bot.command(help="Blackjack. Ex: !blackjack 100 | !blackjack all")
async def blackjack(ctx, amount: str = None):
    if not db_is_ready():
        await ctx.reply("⏳ Databasen startar fortfarande, testa igen om några sekunder.")
        return

    if amount is None:
        await ctx.reply("Ex: `!blackjack 100` | `!blackjack all`")
        return

    key = (ctx.channel.id, ctx.author.id)
    if key in BJ_ACTIVE:
        await ctx.reply("Du har redan en blackjack igång här. Skriv `bj hit` eller `bj stand` reet")
        return
    BJ_ACTIVE.add(key)

    try:
        bal, _ = await econ_get(ctx.author.id)

        if amount.lower() == "all":
            if bal <= 0:
                await ctx.reply("Du har 0 coins. Claim `!daily` eller be någon snäll själ.")
                return
            bet_amount = min(bal, MAX_BET)
        else:
            try:
                bet_amount = int(amount)
            except ValueError:
                await ctx.reply("Amount måste vara ett tal eller `all`.")
                return
            if bet_amount <= 0:
                await ctx.reply("Amount måste vara > 0.")
                return
            if bet_amount > MAX_BET:
                await ctx.reply(f"Max bet är **{MAX_BET}**.")
                return

        # take stake
        new_bal_after_withdraw = await econ_try_withdraw(ctx.author.id, bet_amount)
        if new_bal_after_withdraw is None:
            await ctx.reply(f"Du har inte tillräckligt (balans: **{bal}**).")
            return

        deck = new_deck()
        player = [draw_card(deck), draw_card(deck)]
        dealer = [draw_card(deck), draw_card(deck)]

        def status_text(hide_dealer: bool = True):
            pv = bj_value(player)
            if hide_dealer:
                return (
                    f"🃏 **Blackjack** (bet {bet_amount})\n"
                    f"Du: {', '.join(player)}  (**{pv}**)\n"
                    f"Dealer: {dealer[0]}, ??\n"
                    f"Skriv `bj hit` eller `bj stand` (30s)"
                )
            dv = bj_value(dealer)
            return (
                f"🃏 **Blackjack** (bet {bet_amount})\n"
                f"Du: {', '.join(player)}  (**{pv}**)\n"
                f"Dealer: {', '.join(dealer)}  (**{dv}**)"
            )

        msg = await ctx.send(status_text(hide_dealer=True))

        # Player turn
        while True:
            pv = bj_value(player)

            # natural blackjack (A + 10)
            if pv == 21 and len(player) == 2:
                break

            if pv > 21:
                break

            def check(m: discord.Message):
                return (
                    m.author.id == ctx.author.id
                    and m.channel.id == ctx.channel.id
                    and m.content.lower().strip() in ("bj hit", "bj stand")
                    )

            try:
                reply = await bot.wait_for("message", timeout=30, check=check)
            except asyncio.TimeoutError:
                # timeout = auto stand
                break

            choice = reply.content.lower().strip()
            if choice == "bj hit":
                player.append(draw_card(deck))
                await msg.edit(content=status_text(hide_dealer=True))
                continue
            else:
                break

        pv = bj_value(player)

        # Dealer turn (reveal)
        if pv <= 21:
            while bj_value(dealer) < 17:
                dealer.append(draw_card(deck))

        dv = bj_value(dealer)

        # Resolve
        result = ""
        payout = 0
        profit = -bet_amount

        # natural blackjack payout 3:2
        if pv == 21 and len(player) == 2 and dv != 21:
            payout = int(bet_amount * 2.5)  # returns stake + 1.5x profit
            profit = payout - bet_amount
            result = "player_blackjack"
        elif pv > 21:
            payout = 0
            profit = -bet_amount
            result = "player_bust"
        elif dv > 21:
            payout = bet_amount * 2
            profit = bet_amount
            result = "dealer_bust"
        elif dv == 21 and len(dealer) == 2 and not (pv == 21 and len(player) == 2):
            payout = 0
            profit = -bet_amount
            result = "dealer_blackjack"
        else:
            if pv > dv:
                payout = bet_amount * 2
                profit = bet_amount
                result = "win"
            elif pv < dv:
                payout = 0
                profit = -bet_amount
                result = "loss"
            else:
                # push: give stake back
                payout = bet_amount
                profit = 0
                result = "push"

        if payout > 0:
            new_bal = await econ_deposit(ctx.author.id, payout)
        else:
            new_bal, _ = await econ_get(ctx.author.id)

        await log_bet(ctx.author.id, "blackjack", bet_amount, payout, profit, result)

        end_text = status_text(hide_dealer=False)
        if profit > 0:
            end_text += f"\n✅ Du vann **+{profit}** | Ny balans: **{new_bal}**"
        elif profit < 0:
            end_text += f"\n❌ Du förlorade **{bet_amount}** | Ny balans: **{new_bal}**"
        else:
            end_text += f"\n🤝 Push (pengarna tillbaka...spring tillbaka) | Ny balans: **{new_bal}**"

        await msg.edit(content=end_text)

    finally:
        BJ_ACTIVE.discard((ctx.channel.id, ctx.author.id))



# ============================
#   STARTA BOTTEN
# ============================

bot.run(token, log_handler=handler, log_level=logging.INFO)

