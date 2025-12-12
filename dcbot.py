import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import asyncio
import random
from typing import Optional
import asyncpg

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
TIMEZONE = ZoneInfo('Europe/Stockholm')
INTERVAL_FILE = 'presence_intervals.json'


def load_intervals():
    try:
        with open(INTERVAL_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_intervals(intervals):
    with open(INTERVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(intervals, f, indent=2)


presence_intervals = load_intervals()
last_status: Optional[discord.Status] = None
last_change_time: datetime | None = None


def add_interval(start_dt: datetime, end_dt: datetime):
    if end_dt <= start_dt:
        return

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=TIMEZONE)
    else:
        start_dt = start_dt.astimezone(TIMEZONE)

    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=TIMEZONE)
    else:
        end_dt = end_dt.astimezone(TIMEZONE)

    presence_intervals.append({'start': start_dt.isoformat(), 'end': end_dt.isoformat()})
    save_intervals(presence_intervals)


def is_online_like(status: discord.Status) -> bool:
    return status in (
        discord.Status.online,
        discord.Status.idle,
        discord.Status.dnd,
        discord.Status.invisible,
    )


# ============================
#   DISCORD INTENTS & BOT
# ============================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True

bot = commands.Bot(command_prefix='!', intents=intents)


# ============================
#   KORTTIDSMINNE FÖR GPT
# ============================
# Nyckel: channel_id (eller user_id om du vill per person)
# Värde: lista av {"role": "...", "content": "..."}
conversation_history: dict[int, list[dict]] = {}

@bot.event
async def on_ready():
    global last_status, last_change_time

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
    else:
        last_change_time = None

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

    if was_online and not is_now_online:
        add_interval(last_change_time, now)

    if not was_online and is_now_online:
        last_change_time = now

    last_status = after.status

@bot.command(help="Säger hej till dig. Mest för debug")
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}!!')

@bot.command(help="Skickar Caramelldansen-lyrics.")
async def dansa(ctx):
    await ctx.send('Du-du-du Hey-yeah-yeah-i-yeah \n' 'Vi undrar, är ni redo att vara med? \n' 'Armarna upp, nu ska ni få se \n' 'Kom igen Vem som helst kan vara med (Vara med) \n' 'Så rör på era fötter, o-a-a-a \n' 'Och vicka era höfter, o-la-la-la \n' 'Gör som vi. Till denna melodi \n' 'O-a, o-a-a \n' 'Dansa med oss, klappa era händer \n' 'Gör som vi gör, ta några steg åt vänster \n' 'Lyssna och lär, missa inte chansen \n' 'Nu är vi här med caramelldansen')

@bot.command(help="Visar exakta onlinetider för hebbe senaste 7 dagarna.")
async def hebbe(ctx):
    """Visar exakta onlinetider (start–slut) per dag senaste 7 dagarna."""
    now = datetime.now(TIMEZONE)
    seven_days_ago = now.date() - timedelta(days=6)

    weekday_names = ['Måndag', 'Tisdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lördag', 'Söndag']

    # dag -> lista av (start, slut)
    per_day: dict[datetime.date, list[tuple[datetime, datetime]]] = {}

    for entry in presence_intervals:
        start = datetime.fromisoformat(entry['start']).astimezone(TIMEZONE)
        end = datetime.fromisoformat(entry['end']).astimezone(TIMEZONE)

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
            resp = requests.post(url, json=payload, headers=headers, timeout=15)

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
            resp = requests.post(url, json=payload, headers=headers, timeout=15)

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
            resp = requests.post(url, json=payload, headers=headers, timeout=20)

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



#   ECONOMY OCH ROULETTE

COINS_FILE = "coins.json"
DAILY_COINS = 100
MAX_BET = DAILY_COINS * 10  # 10x daily

economy_lock = asyncio.Lock()
rng = random.SystemRandom()

def load_coins():
    try:
        with open(COINS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_coins(data):
    with open(COINS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

economy = load_coins()

def get_user(econ: dict, user_id: int):
    uid = str(user_id)
    if uid not in econ:
        econ[uid] = {"balance": 0, "last_daily": None}
    return econ[uid]

# Roulette helpers (European roulette: 0-36, 0 is green)
RED_NUMS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK_NUMS = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}

def roulette_color(n: int) -> str:
    if n == 0:
        return "green"
    return "red" if n in RED_NUMS else "black"

def color_emoji(color: str) -> str:
    return {"red": "🔴", "black": "⚫", "green": "🟢"}[color]




#   DAILY FUNC SOM GET 100c
@bot.command(help="Få 100 coins var 24:e timme.")
async def daily(ctx):
    now = datetime.now(TIMEZONE)

    async with economy_lock:
        user = get_user(economy, ctx.author.id)

        if user["last_daily"]:
            last = datetime.fromisoformat(user["last_daily"]).astimezone(TIMEZONE)
            if now - last < timedelta(hours=24):
                remaining = timedelta(hours=24) - (now - last)
                hours, rem = divmod(int(remaining.total_seconds()), 3600)
                mins = (rem // 60)
                await ctx.reply(f"Du har redan claimat daily för fan. Kom tillbaka om **{hours}h {mins}m**.")
                return

        user["balance"] += DAILY_COINS
        user["last_daily"] = now.isoformat()
        save_coins(economy)
        bal = user["balance"]

    await ctx.reply(f"✅ {ctx.author.mention} fick **{DAILY_COINS} coins**! Ny balante: **{bal}** 💰")


# BALANCE COMMAND
@bot.command(aliases=["bal"], help="Visar din coin-balans.")
async def balance(ctx, member: discord.Member | None = None):
    member = member or ctx.author
    async with economy_lock:
        user = get_user(economy, member.id)
        bal = user["balance"]
    await ctx.reply(f"💰 {member.display_name} har **{bal} coins**.")


# LEADERBOARD COMMAND
@bot.command(help="Visar top 10 rikaste spelarna i servern.")
async def leaderboard(ctx):
    async with economy_lock:
        # Build list of (member, balance) only for current guild members
        rows = []
        for uid, data in economy.items():
            m = ctx.guild.get_member(int(uid))
            if m is None:
                continue
            rows.append((m, int(data.get("balance", 0))))

    if not rows:
        await ctx.reply("Ingen har coins än walla Skriv `!daily` för att starta economy.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:10]

    embed = discord.Embed(title="🏆 Leaderboard (Top 10)", description="Mest cash på servern", color=0xF1C40F)
    lines = []
    for i, (m, bal) in enumerate(top, start=1):
        lines.append(f"**#{i}** {m.display_name} — **{bal}** 💰")

    embed.add_field(name="Ranking", value="\n".join(lines), inline=False)
    embed.set_footer(text=f"Max bet: {MAX_BET} | Daily: {DAILY_COINS}/24h")

    await ctx.send(embed=embed)



#BET COMMAND
#endast red black green
#max bet 1000
#cooldown är 1bet / 3 sec
@bot.command(help="Bettar coins på red, black eller green. Ex: !bet 100 red eller !bet all green")
@commands.cooldown(1, 3, commands.BucketType.user)
async def bet(ctx, amount: str = None, color: str = None):
    if amount is None or color is None:
        await ctx.reply("Ex: `!bet 100 red` | `!bet all black` | `!bet 50 green`")
        return

    color = color.lower().strip()
    if color not in ("red", "black", "green"):
        await ctx.reply("Du kan bara betta på: `red`, `black`, `green`.")
        return

    # Resolve amount
    async with economy_lock:
        user = get_user(economy, ctx.author.id)
        bal = int(user["balance"])

    if amount.lower() == "all":
        if bal <= 0:
            await ctx.reply("Du har **0 coins**. Claim `!daily` först kanske")
            return
        bet_amount = min(bal, MAX_BET)  # all-in but still capped by MAX_BET
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

    # Check funds + deduct stake upfront
    async with economy_lock:
        user = get_user(economy, ctx.author.id)
        if user["balance"] < bet_amount:
            await ctx.reply(f"Du har bara **{user['balance']} coins**. Fattig")
            return
        user["balance"] -= bet_amount
        save_coins(economy)

    # Spin animation
    spin_msg = await ctx.send(f"🎰 {ctx.author.mention} bettar **{bet_amount}** på {color_emoji(color)} **{color}**...")
    for _ in range(6):
        fake_num = rng.randrange(0, 37)
        fake_col = roulette_color(fake_num)
        await spin_msg.edit(content=f"🎰 Snurrar... {color_emoji(fake_col)} **{fake_num}**")
        await asyncio.sleep(0.35)

    # Final result
    result_num = rng.randrange(0, 37)
    result_col = roulette_color(result_num)

    # Payouts (profit-based)
    # red/black: 1:1 profit (win returns 2x total including stake)
    # green: 14:1 profit (win returns 15x total including stake)
    if color in ("red", "black"):
        win_total_return = bet_amount * 2
        profit = bet_amount
    else:
        win_total_return = bet_amount * 15
        profit = bet_amount * 14

    win = (result_col == color)

    async with economy_lock:
        user = get_user(economy, ctx.author.id)
        if win:
            user["balance"] += win_total_return
        save_coins(economy)
        new_bal = int(user["balance"])

    if win:
        extra = " (ALL-IN )" if all_in else ""
        await spin_msg.edit(content=(
            f"🎯 RESULTAT: {color_emoji(result_col)} **{result_num}**\n"
            f"✅ {ctx.author.mention} vann{extra}! +**{profit}** profit\n"
            f"💰 Ny balans: **{new_bal}**"
        ))
    else:
        await spin_msg.edit(content=(
            f"🎯 RESULTAT: {color_emoji(result_col)} **{result_num}**\n"
            f"❌ {ctx.author.mention} förlorade **{bet_amount}** coins rip\n"
            f"💰 Ny balans: **{new_bal}**"
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




# ============================
#   STARTA BOTTEN
# ============================
bot.run(token, log_handler=handler, log_level=logging.DEBUG)
