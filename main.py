import os
import re
import pandas as pd
from dotenv import load_dotenv
from google import genai
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardRemove
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction
from telegram.helpers import escape_markdown
from io import BytesIO

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

client = genai.Client(api_key=GEMINI_API_KEY)

user_histories = {}
user_files = {}
user_dataset_types = {}
user_suggestions_shown = {}
user_last_report = {}
user_last_options = {}

def clean_pandas_output(obj):
    if isinstance(obj, pd.Series):
        return obj.to_string(index=True, header=False)
    if isinstance(obj, pd.DataFrame):
        return obj.to_string(index=False)
    return str(obj)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()

def add_user_message(message, history):
    if not message.strip():
        return "", history
    return "", history + [{"role": "user", "content": message}]

def get_bot_response(history, file):
    message = history[-1]['content']

    # if file is None:
    #     intro_message = "📂 Now please upload your CSV file by tapping the 📎 icon below and selecting ‘File’.",
    #     history.append({"role": "assistant", "content": intro_message})
    #     return history

    df = load_dataframe(file)

    try:
        describe_data = df.describe(include="all", datetime_is_numeric=True).fillna("").to_dict()
    except TypeError:
        describe_data = df.describe(include="all").fillna("").to_dict()

    column_summaries = {}
    for col in df.columns:
        if df[col].dtype == "object":
            column_summaries[col] = df[col].value_counts().head(5).to_dict()
        else:
            column_summaries[col] = {
                "min": clean_pandas_output(df[col].min()),
                "max": clean_pandas_output(df[col].max()),
                "mean": clean_pandas_output(df[col].mean())
            }

    schema_info = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape,
        "sample_head": df.head(5).to_dict(orient="records"),
        "sample_tail": df.tail(5).to_dict(orient="records"),
        "sample_random": df.sample(min(5, len(df))).to_dict(orient="records"),
        "describe": describe_data,
        "column_summaries": column_summaries
    }

    code_prompt = f"""
    You are a Python/pandas expert. Based on the uploaded CSV file,
    the DataFrame is called df with columns: {list(df.columns)}.
    Dataset info: {schema_info}

    Write Python code that computes the answer to this question
    and assigns it to a variable named ANSWER.

    Rules:
    - Use only df and pd.
    - Do not import anything.
    - Do not print anything.
    - Return ONLY executable Python code.
    - If question unsupported, set ANSWER = None.

    Question: "{message}"
    """
    code_resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=code_prompt
    )
    generated_code = _extract_code(code_resp.text)

    local_vars = {}
    try:
        exec(generated_code, {"df": df, "pd": pd}, local_vars)
        answer = local_vars.get("ANSWER")
        if isinstance(answer, (pd.Series, pd.DataFrame)):
            answer = clean_pandas_output(answer)
    except Exception:
        answer = None

    if answer is None:
        nlg_prompt = f"""
        The user asked: "{message}".
        The dataset does not contain enough information.
        Reply in 1–2 friendly sentences explaining why.
        """
    else:
        nlg_prompt = f"""
        The user asked: "{message}". Based on the uploaded CSV,
        The computed answer is: {repr(answer)}.
        Summarize the result in 5-10 sentences.
        Reply in a clear, professional tone, style. 
        Dont't use Hello,Hi, or Greetings. Don't mention the dataframe name. 
        Dont't mention code. Dont't say "as an AI model".
        Dont't repeat the question. Dont't apologize. Dont't say "the answer is". Dont't use hastags and emojis.
        """
    nlg_resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=nlg_prompt
    )
    history.append({"role": "assistant", "content": nlg_resp.text.strip()})
    return history

def classify_dataset(file_path: str) -> str:
    df = load_dataframe(file_path)
    schema_info = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape,
    }
    prompt = f"""
    You are a smart classifier. Look at this dataset info: {schema_info}.
    Decide if this dataset is one of:
    - employee_data (HR/people info)
    - sales_data (transactions, revenue, customers)
    - inventory_data (products, stock, warehouses)
    Answer with ONLY one of: employee_data, sales_data, inventory_data, unknown.
    """
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    answer = resp.text.strip().lower()
    if "employee" in answer:
        return "employee_data"
    elif "sales" in answer:
        return "sales_data"
    elif "inventory" in answer:
        return "inventory_data"
    return "unknown"

DATASET_OPTIONS = {
    "employee_data": [
        ["Employee Demographics Report", "Salary by Role"],
        ["Employee Tenure Report", "Headcount Report"],
        ["Salary Distribution Report", "New Hires Report"],
        ["Employee Attrition Report", "Employee Compensation Report"]
    ],
    "sales_data": [
        ["Daily Sales Report", "Product Performance Report"],
        ["Salesperson Performance Report", " Customer Sales Report"],
        ["Store Performance Report", "Transaction Details Report"],
        ["Sales by Product Category Report", "Tax Report"]
    ],
    "inventory_data": [
        ["Stock Levels", "Low Inventory"],
        ["Restock Dates", "Warehouse Summary"],
        ["Fast Movers", "Slow Movers"],
        ["Out of Stock", "Category Breakdown"]
    ],
    "unknown": [
        ["Show Columns", "Basic Stats"],
        ["Row Count", "Data Types"],
        ["Preview Head", "Preview Tail"],
        ["Random Sample", "Value Counts"]
    ]
}

def generate_options(file_path: str, dataset_type: str,  show_suggestions=False, last_button_text=None) -> InlineKeyboardMarkup:
 
    fallback = DATASET_OPTIONS.get(dataset_type, DATASET_OPTIONS["unknown"])
    keyboard = [[InlineKeyboardButton(opt, callback_data=opt) for opt in row] for row in fallback]
    
    if show_suggestions:
        if last_button_text:
            keyboard.append([
                InlineKeyboardButton(f"💡 Suggestions about {last_button_text}", callback_data="suggestions")
            ])
        else:
            keyboard.append([InlineKeyboardButton("💡 Suggestions", callback_data="suggestions")])
    
    keyboard.append([InlineKeyboardButton("📂 Upload a New File", callback_data="upload_new_file")])
    return InlineKeyboardMarkup(keyboard)

# -------------------------
# Telegram Handlers
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Sales Data", callback_data="choose_sales")],
        [InlineKeyboardButton("👥 Employee Data", callback_data="choose_employee")],
        [InlineKeyboardButton("📦 Inventory Data", callback_data="choose_inventory")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🤖 Welcome to Dynamic Data Analyst!\n\n"
        "Choose the dataset type you want to analyze 👇",
        reply_markup=reply_markup
    )

async def help_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Available commands:\n\n"
        "/start – Restart the bot and choose dataset type\n"
        "/help – Show this help message\n"
        "/about – Learn about this bot\n\n"
        "👉 Tip: After uploading a file, please use the option buttons instead of typing."
    )

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Dynamic Data Analyst Bot*\n\n"
        "Upload a CSV or EXCEL file, and I’ll help you analyze it with smart options.\n"
        "Developed by: *ZayMoeYan* ",
        parse_mode="Markdown"
    )



async def handle_dataset_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.message.chat_id
    choice = query.data
    context.user_data["dataset_type"] = choice

    dataset_type_map = {
        "choose_sales": "sales_data",
        "choose_employee": "employee_data",
        "choose_inventory": "inventory_data"
    }

    dataset_type = dataset_type_map.get(choice, "unknown")
    user_dataset_types[user_id] = dataset_type

    await query.message.reply_text(
        f"✅ You selected *{dataset_type.capitalize()} Data*.\n\n"
        "📂 Now please upload your CSV or EXCEL file by tapping the 📎 icon below and selecting ‘File’.",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove()
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    document = update.effective_message.document
    filename = document.file_name.lower()

    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        await update.effective_message.reply_text("❌ Please upload a CSV or Excel file (.csv, .xlsx, .xls).")
        return

    selected_type = user_dataset_types.get(user_id)
    
    if selected_type is None:
        await update.effective_message.reply_text(
            "⚠️ Please select a dataset type first (📊 Sales, 👥 Employee, 📦 Inventory) "
            "before uploading a file."
        )
        return

    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)
    await update.effective_message.reply_text("⏳ Processing file... Please wait.")

    file = await document.get_file()
    file_path = f"/tmp/{user_id}_{document.file_name}"
    await file.download_to_drive(file_path)

    detected_type = classify_dataset(file_path)

    if detected_type != selected_type:
        await update.effective_message.reply_text(
            f"❌ The uploaded file seems to be *{detected_type.replace('_',' ').title()}*, "
            f"but you selected *{selected_type.replace('_',' ').title()}*.\n\n"
            "👉 Please upload the correct file or restart with /start."
        )
        return

    user_files[user_id] = file_path
    # user_dataset_types[user_id] = detected_type
    # markup = generate_options(file_path, detected_type)
    user_suggestions_shown[user_id] = False
    markup = generate_options(file_path, detected_type, show_suggestions=False)


    await update.effective_message.reply_text(
        f"✅ File uploaded and classified as *{escape_markdown(detected_type, version=2)}*"
        "Choose an option below to explore 👇",
        parse_mode="MarkdownV2",
        reply_markup=markup
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    file_path = user_files.get(user_id)

    # If user hasn't uploaded a file yet
    if file_path is None:
        await update.message.reply_text(
            "📂 Please upload a CSV or EXCEL file first by tapping the 📎 icon below."
        )
        return

    # If file is uploaded, but user types a message instead of using buttons
    await update.message.reply_text(
        "ℹ️ Please use the buttons below to explore the dataset.\n"
        "If you want to start over, tap '📂 Upload a New File'."
    )

MAX_TELEGRAM_MESSAGE = 4096
MAX_SPLIT_LIMIT = 20000

def split_message(text, max_len=MAX_TELEGRAM_MESSAGE):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]


async def send_as_file(chat_id, text, context, filename="report.txt"):
    """Send long text as a file to Telegram."""
    bio = BytesIO()
    bio.write(text.encode("utf-8"))
    bio.seek(0)
    await context.bot.send_document(
        chat_id=chat_id,
        document=bio,
        filename=filename
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.message.chat.id

    if query.data == "upload_new_file":
        return await handle_upload_new_file(update, context)

    file_path = user_files.get(user_id)
    dataset_type = user_dataset_types.get(user_id, "unknown")
    history = user_histories.get(user_id, [])
    button_text = query.data
    context.user_data["last_button_text"] = button_text 
    query_text = f"[Dataset type: {dataset_type}] {button_text}"
    _, updated_history = add_user_message(query_text, history)

    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)
    summarizing_msg = f"⏳ Summarizing the report for *{button_text}*... Please wait."
    await query.message.reply_text(summarizing_msg, parse_mode="Markdown")

    bot_response_history = get_bot_response(updated_history, file_path)
    user_histories[user_id] = bot_response_history
    last_reply = bot_response_history[-1]["content"]
 
    user_last_report[user_id] = last_reply

    user_last_options[user_id] = generate_options(
        file_path, dataset_type, show_suggestions=True, last_button_text=button_text
    )

    
    markup = generate_options(file_path, dataset_type, show_suggestions=True)
    # if not user_suggestions_shown.get(user_id, False):
    #     markup = generate_options(file_path, dataset_type, show_suggestions=True)
    #     user_suggestions_shown[user_id] = True


    if len(last_reply) <= MAX_TELEGRAM_MESSAGE:
        await query.message.reply_text(last_reply, reply_markup=markup)
    elif len(last_reply) <= MAX_SPLIT_LIMIT:
        chunks = split_message(last_reply)
        for i, chunk in enumerate(chunks):
            await query.message.reply_text(
                chunk,
                reply_markup=markup if i == len(chunks) - 1 else None
            )
    else:
        await query.message.reply_text("📄 The report is too long. Sending as a file...")
        await send_as_file(user_id, last_reply, context)
        await query.message.reply_text("✅ Report sent as file.", reply_markup=markup)

async def handle_suggestions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.message.chat.id

    last_report = user_last_report.get(user_id)
    last_options = user_last_options.get(user_id)
    button_text = context.user_data.get("last_button_text", "the selected report")  # Retrieve last option

    if not last_report or not last_options:
        await query.message.reply_text("⚠️ No previous report found to base suggestions on.")
        return

    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

    await query.message.reply_text(
        f"💡 Generating suggestions about *{button_text}*. Please wait...",
        parse_mode="Markdown"
    )

    suggestion_prompt = f"""
    You are a business/data analyst. Based on the following report:
    {last_report}

    Provide 5 concise, actionable suggestions and recommendations for the user about {button_text}.
    Format them clearly as bullet points (using "-" at the start of each line).
    Keep it professional and easy to read. 
    Don't repeat the report, don't apologize, don't mention code or dataframes. 
    Don't use greetings, hashtags, or emojis.
    """

    suggestion_resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=suggestion_prompt
    )

    suggestions_text = suggestion_resp.text.strip()

    # Ensure bullet formatting (if Gemini doesn’t output dashes, we fix it)
    lines = suggestions_text.splitlines()
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("-"):
            line = f"- {line}"
        formatted_lines.append(line)
    formatted_text = "\n".join(formatted_lines)

    # Final wrapped message
    final_suggestions = f"📌 *Suggestions about {button_text}:*\n\n{formatted_text}"

    await query.message.reply_text(
        final_suggestions, 
        parse_mode="Markdown", 
        reply_markup=generate_options(file_path=None, dataset_type=user_dataset_types[user_id], show_suggestions=True)
    )



async def handle_upload_new_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.message.chat.id

    # Clear previous data
    user_files.pop(user_id, None)
    user_histories.pop(user_id, None)
    user_dataset_types.pop(user_id, None)

    keyboard = [
        [InlineKeyboardButton("📊 Sales Data", callback_data="choose_sales")],
        [InlineKeyboardButton("👥 Employee Data", callback_data="choose_employee")],
        [InlineKeyboardButton("📦 Inventory Data", callback_data="choose_inventory")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text(
        "🤖 Let's start over!\n\n"
        "Choose the dataset type you want to analyze 👇",
        reply_markup=reply_markup
    )

# -------------------------
# Main Entry
# -------------------------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_func))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CallbackQueryHandler(handle_dataset_choice, pattern="^choose_"))
    app.add_handler(CallbackQueryHandler(handle_suggestions, pattern="^suggestions$"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Button handler: all except upload_new_file
    app.add_handler(CallbackQueryHandler(handle_button, pattern="^(?!upload_new_file).+"))
    # Upload new file handler
    app.add_handler(CallbackQueryHandler(handle_upload_new_file, pattern="^upload_new_file$"))

    print("🚀 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
