import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests  # används för DeepSeek-anropet

# ============================
#   LÄS MILJÖVARIABLER
# ============================
load_dotenv()
token = os.getenv("DISCORD_TOKEN")

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
print("DEBUG - DEEPSEEK_API_KEY =", "SATT" if deepseek_key else "SAKNAS")

if not token:
    print("VARNING: DISCORD_TOKEN saknas i miljövariablerna!")
if not deepseek_key:
    print("VARNING: DEEPSEEK_API_KEY saknas i miljövariablerna!")

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


# ============================
#   KOMMANDON
# ============================
@bot.command()
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}!!')

@bot.command()
async def dansa(ctx):
    await ctx.send('Du-du-du Hey-yeah-yeah-i-yeah \n' 'Vi undrar, är ni redo att vara med? \n' 'Armarna upp, nu ska ni få se \n' 'Kom igen Vem som helst kan vara med (Vara med) \n' 'Så rör på era fötter, o-a-a-a \n' 'Och vicka era höfter, o-la-la-la \n' 'Gör som vi. Till denna melodi \n' 'O-a, o-a-a \n' 'Dansa med oss, klappa era händer \n' 'Gör som vi gör, ta några steg åt vänster \n' 'Lyssna och lär, missa inte chansen \n' 'Nu är vi här med caramelldansen')


# ============================
#   DEEPSEEK GPT-FUNKTION
# ============================
@bot.command(name="gpt")
async def gpt(ctx, *, prompt: str | None = None):
    """Skickar prompten till DeepSeek och svarar med svaret."""
    print(f"!gpt triggat av {ctx.author} med prompt: {prompt!r}")

    if not prompt:
        await ctx.reply("Du måste skriva något efter kommandot idiot, t.ex. `!gpt skriv en dikt om 67`")
        return

    await ctx.trigger_typing()

    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_KEY:
        await ctx.reply("DeepSeek-nyckel saknas (DEEPSEEK_API_KEY). Kolla Railway Variables.")
        return

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Du är en hjälpsam svensk assistent i en Discord-server. Svara som om du är en kroniskt online meme-besatt ungdom."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "stream": False,
    }

    print("[DeepSeek] Skickar request ...")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
    except Exception as e:
        print("[DeepSeek] Nätverksfel:", repr(e))
        await ctx.reply(f"Nätverksfel mot DeepSeek:\n`{e}`")
        return

    print("[DeepSeek] Fick svar, statuskod:", resp.status_code)
    print("[DeepSeek] Rått svar (första 400 tecken):", resp.text[:400])

    if resp.status_code != 200:
        await ctx.reply(f"DeepSeek API error {resp.status_code}:\n```text\n{resp.text[:300]}\n```")
        return

    try:
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        print("[DeepSeek] JSON-/formatfel:", repr(e))
        await ctx.reply("Kunde inte tolka svaret från DeepSeek.")
        return

    if not reply:
        reply = "Jag fick ett tomt svar från modellen."

    if len(reply) <= 2000:
        await ctx.reply(reply)
    else:
        for i in range(0, len(reply), 1900):
            await ctx.send(reply[i:i+1900])



# ====== KÖR BOTTEN ======
bot.run(token, log_handler=handler, log_level=logging.DEBUG)
