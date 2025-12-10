import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

# ============================
#   LÄS MILJÖVARIABLER
# ============================
load_dotenv()
token = os.getenv("DISCORD_TOKEN")
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
last_status: discord.Status | None = None
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

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}!!')

@bot.command()
async def dansa(ctx):
    await ctx.send('Du-du-du Hey-yeah-yeah-i-yeah \n' 'Vi undrar, är ni redo att vara med? \n' 'Armarna upp, nu ska ni få se \n' 'Kom igen Vem som helst kan vara med (Vara med) \n' 'Så rör på era fötter, o-a-a-a \n' 'Och vicka era höfter, o-la-la-la \n' 'Gör som vi. Till denna melodi \n' 'O-a, o-a-a \n' 'Dansa med oss, klappa era händer \n' 'Gör som vi gör, ta några steg åt vänster \n' 'Lyssna och lär, missa inte chansen \n' 'Nu är vi här med caramelldansen')

@bot.command()
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
            parts = [f"{start.strftime("%H:%M")}-{end.strftime("%H:%M")}" for start, end in intervals]
            lines.append(f"{weekday}: " + " & ".join(parts))

    msg = '**Onlinetider (senaste 7 dagarna):**\n```text\n' + '\n'.join(lines) + '\n```'
    await ctx.send(msg)


# ============================
#   GPT / GROQ-KOMMANDO
# ============================
@bot.command(name="gpt")
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

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Du är en snabb och hjälpsam Discord-bot. "
                    "Svara kort, roligt och ungt. Släng gärna in lite meme / internet slang."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }

    print("[Groq] Förbereder request...")

    try:
        # Visa "botten skriver..." medan vi gör API-anropet
        async with ctx.channel.typing():
            print("[Groq] Skickar request...")
            resp = requests.post(url, json=payload, headers=headers, timeout=15)

        print("[Groq] Statuskod:", resp.status_code)
        print("[Groq] Förhandsvisning av svar:", resp.text[:300])

        if resp.status_code != 200:
            await ctx.reply(f"Groq API-fel {resp.status_code}:\n```{resp.text[:300]}```")
            return

        # Tolka JSON-svaret
        try:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
        except Exception as e:
            print("[Groq] JSON-fel:", repr(e))
            await ctx.reply("Kunde inte tolka svaret från Groq.")
            return

        if not reply:
            reply = "Jag fick ett tomt svar walla"

        # Skicka svaret (hantera långa)
        if len(reply) <= 2000:
            await ctx.reply(reply)
        else:
            for i in range(0, len(reply), 1900):
                await ctx.send(reply[i:i+1900])

    except Exception as e:
        print("[Groq] Oväntat fel i gpt-kommandot:", repr(e))
        await ctx.reply(f"Något gick snett i gpt-kommandot:\n`{repr(e)}`")



# ============================
#   STARTA BOTTEN
# ============================
bot.run(token, log_handler=handler, log_level=logging.DEBUG)
