import os
import logging
import requests
from dotenv import load_dotenv
import discord
from discord.ext import commands

# ========= ENV =========
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

print("DEBUG DISCORD_TOKEN =", "SATT" if DISCORD_TOKEN else "SAKNAS")
print("DEBUG DEEPSEEK_API_KEY =", "SATT" if DEEPSEEK_API_KEY else "SAKNAS")

# ========= LOGGING =========
handler = logging.FileHandler(filename='dcbot.log', encoding='utf-8', mode='a')

# ========= DISCORD BOT =========
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


@bot.command()
async def hello(ctx):
    await ctx.send(f"Hello {ctx.author.mention}!!")


@bot.command(name="gpt")
async def gpt(ctx, *, prompt: str | None = None):
    """Skickar prompten till DeepSeek och svarar."""
    print(f"!gpt triggat av {ctx.author} med prompt: {prompt!r}")

    if not prompt:
        await ctx.reply("Du måste skriva något efter kommandot, t.ex. `!gpt skriv en dikt om 67`")
        return

    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_KEY:
        await ctx.reply("DEEPSEEK_API_KEY saknas i env/Variables.")
        return

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Du är en hjälpsam svensk Discord-bot."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "stream": False,
    }

    try:
        print("[GPT] Går in i huvud-try-blocket")

        # Visa "botten skriver..." medan vi gör API-anropet
        async with ctx.channel.typing():
            print("[DeepSeek] Skickar request...")
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            print("[DeepSeek] Statuskod:", resp.status_code)
            print("[DeepSeek] Rått svar (början):", resp.text[:300])

            if resp.status_code != 200:
                await ctx.reply(
                    f"DeepSeek API-fel {resp.status_code}:\n```text\n{resp.text[:300]}\n```"
                )
                return

            try:
                data = resp.json()
                reply = data["choices"][0]["message"]["content"]
            except Exception as e:
                print("[DeepSeek] JSON/format-fel:", repr(e))
                await ctx.reply("Kunde inte tolka svaret från DeepSeek.")
                return

        # (utanför typing-blocket)
        if not reply:
            reply = "Jag fick ett tomt svar från modellen."

        if len(reply) <= 2000:
            await ctx.reply(reply)
        else:
            for i in range(0, len(reply), 1900):
                await ctx.send(reply[i:i+1900])

    except Exception as e:
        print("[GPT] Fångade undantag i gpt-kommandot:", repr(e))
        await ctx.reply(f"Internt fel i gpt-kommandot:\n`{repr(e)}`")


# ========= STARTA BOTTEN =========
if not DISCORD_TOKEN:
    print("FATAL: DISCORD_TOKEN saknas, kan inte starta botten.")
else:
    bot.run(DISCORD_TOKEN, log_handler=handler, log_level=logging.DEBUG)
