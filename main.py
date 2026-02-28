import os
import pyodbc
import pandas as pd
import numpy as np
from datetime import date, datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv

from ai_logic import answer_ceo_question  # AI logic

# .env load (DB_PASSWORD, OPENAI_API_KEY, etc.)
load_dotenv()

app = FastAPI(title="CEO AI Backend")

origins = ["http://ceodashboard.softgoway.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pyodbc.pooling = True


def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=103.171.180.23;"
        "DATABASE=CEO_DASH;"
        "UID=CEO_DASH;"
        f"PWD={os.getenv('DB_PASSWORD')};"
        "TrustServerCertificate=yes;"
    )


EMPLOYEES_TABLE = "employees"
FORMER_EMPLOYEES_TABLE = "former_employees"
INSTRUCTORS_TABLE = "instructor_performance"
NEW_JOINERS_TABLE = "new_joiners_2023_2024"


def df_to_list(df: pd.DataFrame):
    return df.to_dict(orient="records")


# --------- ADDED: Department breakdown helper ---------

def build_department_breakdown(employees_df: pd.DataFrame, former_df: pd.DataFrame) -> list:
    """
    Build per-department summary: active count, former count, payroll, attrition rate.
    Returns a JSON-safe list of dicts.
    """
    if employees_df.empty or "department" not in employees_df.columns:
        return []

    dept_counts = employees_df["department"].dropna().value_counts()
    breakdown = []

    for dept_name, active_cnt in dept_counts.items():
        former_cnt = 0
        if not former_df.empty and "department" in former_df.columns:
            former_cnt = int((former_df["department"] == dept_name).sum())

        monthly_pay = 0.0
        if "total_salary" in employees_df.columns:
            monthly_pay = float(
                employees_df[employees_df["department"] == dept_name]["total_salary"]
                .fillna(0).replace([np.inf, -np.inf], 0).sum()
            )
            if not np.isfinite(monthly_pay):
                monthly_pay = 0.0

        total_in_dept = int(active_cnt) + former_cnt
        dept_attrition = round((former_cnt / total_in_dept) * 100, 1) if total_in_dept > 0 else 0.0

        breakdown.append({
            "department": str(dept_name),
            "active_count": int(active_cnt),
            "former_count": former_cnt,
            "monthly_payroll_qr": round(monthly_pay, 2),
            "attrition_rate": dept_attrition,
        })

    return breakdown


# ---------- Root health-check ----------

@app.get("/")
async def root():
    return {"status": "ok", "message": "CEO AI Backend running"}


# ---------- HR dashboard endpoint ----------

@app.get("/api/hr-dashboard")
async def hr_dashboard():
    """
    Returns summary + employees + new joiners + instructors + former employees.
    """
    try:
        conn = get_connection()
        employees_df = pd.read_sql(f"SELECT * FROM [{EMPLOYEES_TABLE}]", conn)
        former_df = pd.read_sql(f"SELECT * FROM [{FORMER_EMPLOYEES_TABLE}]", conn)
        instructor_df = pd.read_sql(f"SELECT * FROM [{INSTRUCTORS_TABLE}]", conn)
        new_joiners_df = pd.read_sql(f"SELECT * FROM [{NEW_JOINERS_TABLE}]", conn)
        conn.close()

        total_employees = len(employees_df)
        total_leavers = len(former_df)
        attrition_rate = (
            round((total_leavers / total_employees) * 100, 1)
            if total_employees
            else 0
        )

        employee_summary = {
            "total_employees": total_employees,
            "active_employees": total_employees,
            "attrition_rate": attrition_rate,
            "engagement_score": 0,
        }

        # DataFrames -> basic clean
        def df_safe(df: pd.DataFrame):
            df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
            for col in df.columns:
                if df[col].dtype == "datetime64[ns]":
                    df[col] = df[col].dt.strftime("%Y-%m-%d")
            return df.to_dict(orient="records")

        # ADDED: department breakdown
        department_breakdown = build_department_breakdown(employees_df, former_df)

        payload = {
            "employee": employee_summary,
            "employees": df_safe(employees_df),
            "new_joiners": df_safe(new_joiners_df),
            "instructors": df_safe(instructor_df),
            "employee_trend": [],
            "former": df_safe(former_df),
            "department_breakdown": department_breakdown,  # ADDED
        }

        safe_content = jsonable_encoder(payload)
        return JSONResponse(content=safe_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Models ----------

class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    reply: str


# ---------- AI chat endpoint (uses ai_logic) ----------

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ceo_ai(body: ChatRequest):
    try:
        conn = get_connection()
        employees_df = pd.read_sql(f"SELECT * FROM [{EMPLOYEES_TABLE}]", conn)
        former_df = pd.read_sql(f"SELECT * FROM [{FORMER_EMPLOYEES_TABLE}]", conn)
        instructor_df = pd.read_sql(f"SELECT * FROM [{INSTRUCTORS_TABLE}]", conn)
        new_joiners_df = pd.read_sql(f"SELECT * FROM [{NEW_JOINERS_TABLE}]", conn)
        conn.close()

        total_employees = len(employees_df)
        total_leavers = len(former_df)
        total_joiners = len(new_joiners_df)

        # ---- Base metrics ----
        attrition_rate = (
            round((total_leavers / total_employees) * 100, 1)
            if total_employees
            else 0.0
        )

        # ---- Turnover cost (30% of annual salary of leavers) ----
        turnover_cost_qr = 0.0

        if total_leavers > 0 and "basic_salary" in former_df.columns:
            if "total_salary" in former_df.columns:
                monthly = former_df["total_salary"].fillna(0)
            else:
                monthly = (
                    former_df["basic_salary"].fillna(0)
                    + former_df.get("housing_allowance", 0).fillna(0)
                    + former_df.get("transport_allowance", 0).fillna(0)
                )

            annual = monthly * 12
            turnover_cost_qr = float((annual * 0.3).sum())
        else:
            turnover_cost_qr = 0.0

        # ADDED: department stats for AI
        department_breakdown_list = build_department_breakdown(employees_df, former_df)
        total_departments = len(department_breakdown_list)
        department_names = [d["department"] for d in department_breakdown_list]

        base_metrics = {
            "total_employees": total_employees,
            "active_employees": total_employees,
            "former_employees": total_leavers,
            "attrition_rate": attrition_rate,
            "turnover_cost_qr": turnover_cost_qr,
            "new_joiners_count": total_joiners,
            # ADDED: department info so AI can answer department questions
            "total_departments": total_departments,
            "department_names": department_names,
            "department_breakdown": department_breakdown_list,
        }

        # ---- Performance metrics (from instructor_performance + probation) ----
        performance_metrics = {
            "poor_performers_count": 0,
            "exceed_expectations_pct": 0.0,
            "top_performer_name": None,
            "probation_failed_count": 0,
            "probation_failed_names": [],
        }

        # instructor_performance: poor / exceed / top performer
        if not instructor_df.empty and "overall_rating" in instructor_df.columns:
            total_reviews = len(instructor_df)

            poor_mask = instructor_df["overall_rating"].str.lower().isin(
                ["below expectations", "needs improvement", "poor performer"]
            )
            performance_metrics["poor_performers_count"] = int(poor_mask.sum())

            good_mask = instructor_df["overall_rating"].str.lower().isin(
                ["exceed expectations"]
            )
            performance_metrics["exceed_expectations_pct"] = (
                round((good_mask.sum() / total_reviews) * 100, 1)
                if total_reviews
                else 0.0
            )

            if "overall_score" in instructor_df.columns:
                sorted_perf = instructor_df.sort_values(
                    by=["overall_score", "review_date"],
                    ascending=[False, False]
                )
                top_row = sorted_perf.iloc[0]
                if "employee_name" in sorted_perf.columns:
                    performance_metrics["top_performer_name"] = top_row["employee_name"]

        # NEW: probation failures from new_joiners_2023_2024
        if not new_joiners_df.empty and "probation_status" in new_joiners_df.columns:
            fail_mask = new_joiners_df["probation_status"].str.lower() == "failed"
            performance_metrics["probation_failed_count"] = int(fail_mask.sum())

            if (
                "first_name" in new_joiners_df.columns
                and "last_name" in new_joiners_df.columns
            ):
                performance_metrics["probation_failed_names"] = [
                    f"{fn} {ln}"
                    for fn, ln in zip(
                        new_joiners_df.loc[fail_mask, "first_name"],
                        new_joiners_df.loc[fail_mask, "last_name"],
                    )
                ]

        # ---- Call AI logic ----
        reply = answer_ceo_question(
            question=body.prompt,
            employees_df=employees_df,
            former_df=former_df,
            base_metrics=base_metrics,
            performance_metrics=performance_metrics,
            new_joiners_df=new_joiners_df,
        )

        return ChatResponse(reply=reply)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))