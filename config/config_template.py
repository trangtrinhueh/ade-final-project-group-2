# ============================================================
#  config/config_template.py  —  Template cấu hình
#  Sao chép thành config/config.py và điền thông tin thực:
#      cp config/config_template.py config/config.py
# ============================================================

SERVER   = "YOUR_SERVER_NAME"
DATABASE = "retail_forecast"
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"

DRIVER = "ODBC Driver 17 for SQL Server"

CONNECTION_STRING = (
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    f"TrustServerCertificate=yes;"
)

STAGING_SALES_RAW = "staging_sales_raw"
STAGING_PRODUCT   = "staging_product"
STAGING_STORE     = "staging_sales_raw"
GEN_PROVINCE      = "gen_province"
PM_MAINGROUP      = "pm_maingroup"
PM_SUBGROUP       = "pm_subgroup"

FACT_TABLE  = "FACT_DAILY_SALES"
DIM_DATE    = "DIM_DATE"
DIM_PRODUCT = "DIM_PRODUCT"
DIM_STORE   = "DIM_STORE"

LAG_DAYS       = 7
TRAIN_END_DAY  = 21
VAL_END_DAY    = 26

LEAD_TIME_DAYS = 1
SAFETY_FACTOR  = 0.15
