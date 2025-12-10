import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from openai import OpenAI, OpenAIError  # <--- BOMBSÄKER IMPORT!
import requests  # för test_openai_api


load_dotenv()
token = os.getenv("DISCORD_TOKEN")

api_key = os.getenv("MY_OPENAI_KEY")
print("DEBUG - MY_OPENAI_KEY =", "SATT if api_key else 'SAKNAS'")

if api_key is None:
    print("VARNING: MY_OPENAI_KEY saknas i miljövariablerna!")

# ====== INITIALLISERA OPENAI KLIENTEN ======
# Klienten är synkron, men vi kör den i en executor i en annan tråd.
openai_client = None
if api_key:
    try:
        # Använder api_key från MY_OPENAI_KEY
        openai_client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"KRITISKT FEL: Kunde inte initialisera OpenAI-klienten. Fel: {e}")


# ====== TESTA OPENAI API VID START ======
def test_openai_api():
    """Snabb diagnostik: kolla om API-nyckeln funkar och vad OpenAI svarar."""
    if not api_key:
        print("❌ OPENAI TEST: Ingen API-nyckel (MY_OPENAI_KEY) hittades.")
        return

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Säg exakt: API fungerar"}],
        "max_tokens": 20,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("\n===== KÖR OPENAI START-TEST =====")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        print("OPENAI TEST – statuskod:", resp.status_code)
        print("OPENAI TEST – rått svar (första 500 tecken):")
        print(resp.text[:500])
        print("===== SLUT PÅ OPENAI START-TEST =====\n")
    except Exception as e:
        print("❌ OPENAI TEST: Kunde inte kontakta API:t.")
        print("Fel:", repr(e))
        print("===== SLUT PÅ OPENAI START-TEST (FEL) =====\n")


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

last_status: discord.Status | None = None
last_change_time: datetime | None = None

def is_online_like(status: discord.Status) -> bool:
    return status in (
        discord.Status.online,
        discord.Status.idle,
        discord.Status.dnd,
        discord.Status.invisible,
    )

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    global last_status, last_change_time

    print(f'Logged in as {bot.user} (ID: {bot.user.id})')

    # Försök hitta din user i alla guilds
    owner_member = None
    for guild in bot.guilds:
        m = guild.get_member(OWNER_ID)
        if m is not None:
            owner_member = m
            break

    now = datetime.now(TIMEZONE)

    if owner_member is None:
        # Om du inte är i samma guild som botten → ingen presence kan hämtas
        print("VARNING: Kunde inte hitta OWNER_ID i någon guild.")
        last_status = discord.Status.offline
        last_change_time = now
        return

    # Sätt status baserat på faktisk status
    current_status = owner_member.status
    last_status = current_status

    # Om du är online → starta en intervall direkt från bot-start
    if is_online_like(current_status):
        last_change_time = now
    else:
        # Om du är offline → ingen aktiv intervall
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



# ====== KÖR BOTTEN ======
bot.run(token, log_handler=handler, log_level=logging.DEBUG)
