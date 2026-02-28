# ai_logic.py

import os
from typing import Dict, Any, List

import pandas as pd
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"  # ya "gpt-4o" agar access hai

# ========= CEO prompt template =========

ceo_prompt = """
You are 'CEO AI', an executive HR & workforce analytics assistant.
You ONLY answer using the numeric data and metrics provided below.
If a metric is missing, say it is not available instead of guessing.

DATA_CONTEXT:
HR_OVERVIEW = {hr_overview}
EMPLOYEES_METRICS = {employees}

RULES:
- Answer as if speaking to the CEO.
- Be concise, 3–6 short sentences.
- Always mention specific numbers (counts, %, amounts) when available.
- Always include: 1) a quick status, 2) any warnings or risks, 3) 1–2 concrete suggestions.
- If data is missing for part of the question, clearly say which data is not available.
- Do NOT explain how you calculated things, only give final insights.
- Do NOT invent numbers that are not in HR_OVERVIEW or EMPLOYEES_METRICS.

QUESTION FROM CEO:
{question}

Now give a clear, executive summary answer with risks and suggested actions.
"""

# ===================== BASE METRIC FALLBACKS ==========================

def fetch_employee_data() -> Dict[str, Any]:
    """
    Fallback metrics agar main.py se kuch na aaye.
    """
    return {
        "total_employees": 0,
        "active_employees": 0,
        "former_employees": 0,
        "attrition_rate": 0.0,
        "monthly_payroll_qr": 0.0,
        "turnover_last_year": 0,
        "turnover_breakdown": {},
        "top_turnover_department": None,
        "top_turnover_pct": 0.0,
        "turnover_main_reasons": [],
        "turnover_cost_qr": 0.0,
        "engagement_score": None,
        "new_joiners_2023_2024": 0,
        "on_probation": 0,
        "top_nationality": None,
        "top_nationality_pct": 0.0,
        "contract_split": {},
        "avg_pass_rate": 0.0,
        "avg_rating": 0.0,
        "top_instructor": None,
        "avg_tenure_years": 0.0,
        "nationality_split": {},
        "married_pct": 0.0,
        "single_pct": 0.0,
    }

# ==================== HR OVERVIEW BUILDER =============================

def build_hr_overview(
    employees_df: pd.DataFrame | None,
    former_df: pd.DataFrame | None,
    base_metrics: Dict[str, Any] | None = None,
    performance_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    HR overview dict jo prompt me inject hoga.
    """
    m = base_metrics or fetch_employee_data()
    p = performance_metrics or {
        "poor_performers_count": 0,
        "exceed_expectations_pct": 0.0,
        "top_performer_name": None,
        "probation_failed_count": 0,
        "probation_failed_names": [],
    }

    employees_df = employees_df if employees_df is not None else pd.DataFrame()
    former_df = former_df if former_df is not None else pd.DataFrame()

    # ---------- Workforce KPIs from employees_df ----------

    monthly_payroll_qr = m.get("monthly_payroll_qr")
    if (monthly_payroll_qr is None or monthly_payroll_qr == 0) and not employees_df.empty:
        if "total_salary" in employees_df.columns:
            monthly_payroll_qr = float(employees_df["total_salary"].sum())
        else:
            monthly_payroll_qr = 0.0

    avg_tenure_years = m.get("avg_tenure_years")
    if (avg_tenure_years is None or avg_tenure_years == 0) and not employees_df.empty:
        if "date_of_joining" in employees_df.columns:
            tmp = employees_df.copy()
            tmp["doj"] = pd.to_datetime(tmp["date_of_joining"])
            tmp["tenure_years"] = (pd.Timestamp("today") - tmp["doj"]).dt.days / 365
            avg_tenure_years = round(tmp["tenure_years"].mean(), 1)
        else:
            avg_tenure_years = 0.0

    nationality_split = m.get("nationality_split") or {}
    if not nationality_split and not employees_df.empty and "nationality" in employees_df.columns:
        nationality_split = (
            employees_df["nationality"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .to_dict()
        )

    married_pct = m.get("married_pct")
    single_pct = m.get("single_pct")
    if (married_pct is None or single_pct is None) and not employees_df.empty:
        if "contract_type" in employees_df.columns:
            ct = (
                employees_df["contract_type"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
            )
            married_pct = float(ct.get("Married", 0))
            single_pct = float(ct.get("Single", 0))
        else:
            married_pct = 0.0
            single_pct = 0.0

    # ---------- Turnover metrics from former_df ----------

    turnover_last_year = m.get("turnover_last_year")
    turnover_breakdown = m.get("turnover_breakdown")
    top_turnover_department = m.get("top_turnover_department")
    top_turnover_pct = m.get("top_turnover_pct")
    turnover_main_reasons = m.get("turnover_main_reasons")

    if not former_df.empty:
        total_leavers = len(former_df)

        if "dateofleaving" in former_df.columns:
            fd = former_df.copy()
            fd["dol"] = pd.to_datetime(fd["dateofleaving"])
            last_year = pd.Timestamp("today").year - 1
            last_year_leavers = fd[fd["dol"].dt.year == last_year]
            turnover_last_year = len(last_year_leavers)
        else:
            turnover_last_year = total_leavers

        termination_count = resignation_count = 0
        if "terminationtype" in former_df.columns:
            termination_count = int((former_df["terminationtype"] == "Termination").sum())
            resignation_count = int((former_df["terminationtype"] == "Resignation").sum())
        turnover_breakdown = {
            "terminations": termination_count,
            "resignations": resignation_count,
            "total_leavers": total_leavers,
        }

        if "department" in former_df.columns:
            dept_counts = former_df["department"].value_counts()
            if not dept_counts.empty:
                top_turnover_department = dept_counts.index[0]
                top_turnover_pct = round(
                    (dept_counts.iloc[0] / total_leavers) * 100, 1
                )

        if "terminationsubreason" in former_df.columns and "terminationtype" in former_df.columns:
            reason_counts = (
                former_df[former_df["terminationtype"] == "Resignation"]
                .groupby("terminationsubreason")
                .size()
                .sort_values(ascending=False)
            )
            turnover_main_reasons = [
                f"{reason}: {count} resignations"
                for reason, count in reason_counts.head(3).items()
            ]

    turnover_cost_qr = m.get("turnover_cost_qr")
    if (
        (turnover_cost_qr is None or turnover_cost_qr == 0)
        and not former_df.empty
        and not employees_df.empty
        and "employee_number" in former_df.columns
        and "employee_number" in employees_df.columns
        and "total_salary" in employees_df.columns
    ):
        sal = employees_df[["employee_number", "total_salary"]].drop_duplicates()
        leavers = former_df.merge(sal, on="employee_number", how="left")
        leavers["annual_salary"] = leavers["total_salary"].fillna(0) * 12
        leavers["turnover_cost_est"] = leavers["annual_salary"] * 0.3
        turnover_cost_qr = float(leavers["turnover_cost_est"].sum())
    elif turnover_cost_qr is None:
        turnover_cost_qr = 0.0

    return {
        "total_employees": m.get("total_employees"),
        "active_employees": m.get("active_employees"),
        "former_employees": m.get("former_employees"),
        "attrition_rate": m.get("attrition_rate"),
        "monthly_payroll_qr": monthly_payroll_qr,
        "turnover_last_year": turnover_last_year,
        "turnover_breakdown": turnover_breakdown,
        "top_turnover_department": top_turnover_department,
        "top_turnover_pct": top_turnover_pct,
        "turnover_main_reasons": turnover_main_reasons,
        "turnover_cost_qr": turnover_cost_qr,
        "engagement_score": m.get("engagement_score"),
        # yahan base_metrics["new_joiners_count"] bhi pick hoga
        "new_joiners_2023_2024": m.get("new_joiners_2024") or m.get("new_joiners_2023_2024") or m.get("new_joiners_count"),
        "on_probation": m.get("on_probation"),
        "top_nationality": m.get("top_nationality"),
        "top_nationality_pct": m.get("top_nationality_pct"),
        "contract_split": m.get("contract_split"),
        "avg_pass_rate": m.get("avg_pass_rate"),
        "avg_rating": m.get("avg_rating"),
        "top_instructor": m.get("top_instructor"),
        "nationality_split": nationality_split,
        "married_pct": married_pct,
        "single_pct": single_pct,
        "avg_tenure_years": avg_tenure_years,
        "poor_performers_count": p.get("poor_performers_count"),
        "exceed_expectations_pct": p.get("exceed_expectations_pct"),
        "top_performer_name": p.get("top_performer_name"),
        "probation_failed_count": p.get("probation_failed_count"),
        "probation_failed_names": p.get("probation_failed_names"),
    }

# ===================== LOOKUP HELPERS =================================

def lookup_term_in_hr_data(
    term: str,
    employees_df: pd.DataFrame | None,
    former_df: pd.DataFrame | None,
) -> Dict[str, Any]:
    term = term.strip().lower()
    summary: Dict[str, Any] = {}
    employees_df = employees_df if employees_df is not None else pd.DataFrame()
    former_df = former_df if former_df is not None else pd.DataFrame()

    if not employees_df.empty and "department" in employees_df.columns:
        dept_mask = employees_df["department"].str.lower() == term
        dept_rows = employees_df[dept_mask]
        if not dept_rows.empty:
            summary["department_name"] = term
            summary["headcount"] = int(len(dept_rows))
            if "total_salary" in dept_rows.columns:
                summary["monthly_payroll_qr"] = float(dept_rows["total_salary"].sum())
            if "job_title" in dept_rows.columns:
                summary["top_roles"] = (
                    dept_rows["job_title"].value_counts().head(5).to_dict()
                )

    if not former_df.empty and "department" in former_df.columns:
        dept_mask_f = former_df["department"].str.lower() == term
        dept_leavers = former_df[dept_mask_f]
        if not dept_leavers.empty:
            summary["leavers_total"] = int(len(dept_leavers))
            if "dateofleaving" in dept_leavers.columns:
                fd = dept_leavers.copy()
                fd["dol"] = pd.to_datetime(fd["dateofleaving"])
                last_year = pd.Timestamp("today").year - 1
                last_year_leavers = fd[fd["dol"].dt.year == last_year]
                summary["leavers_last_year"] = int(len(last_year_leavers))
            if "terminationtype" in dept_leavers.columns:
                summary["termination_breakdown"] = (
                    dept_leavers["terminationtype"].value_counts().to_dict()
                )

    return summary
#===========================================gelberish===========================================
def is_gibberish(text: str) -> bool:
        text = text.strip()

        # Case 1: only letters, no spaces, length >= 6 → very likely nonsense like "jbdshkfcag"
        if re.fullmatch(r"[A-Za-z]{4,}", text):
            return True

        # Case 2: no letters at all (only digits/symbols) → gibberish
        letters = len(re.findall(r"[A-Za-z]", text))
        if letters == 0:
            return True

        return False        

# ===================== MAIN CEO ANSWER FUNC ===========================

NATIONALITY_KEYWORDS = [
    "nationality", "nationalities", "indian", "qatari", "qataris",
    "nepali", "filipino", "egyptian", "pakistani", "sudanese",
    "british", "french", "malaysian", "jordanian", "romanian", "spanish",
    "italian", "vietnamese", "palestinian", "somali", "chinese",
]

def answer_ceo_question(
    question: str,
    employees_df: pd.DataFrame | None = None,
    former_df: pd.DataFrame | None = None,
    base_metrics: Dict[str, Any] | None = None,
    performance_metrics: Dict[str, Any] | None = None,
    new_joiners_df: pd.DataFrame | None = None,
) -> str:
    
    
    q_raw = question.strip()
    # Step 1: gibberish guard
    if is_gibberish(q_raw):
        return (
            "I did not understand this request. "
            "Please ask a clear HR-related question, such as about employees, departments, "
            "attrition, new joiners, or workforce statistics."
        )

    # If not gibberish, continue with your existing logic
    q = q_raw.lower()
    employees_df = employees_df if employees_df is not None else pd.DataFrame()
    former_df = former_df if former_df is not None else pd.DataFrame()
    new_joiners_df = new_joiners_df if new_joiners_df is not None else pd.DataFrame()

    # 0) Department / term direct lookup
    words = [w.strip(".,!? ") for w in q.split() if len(w) > 1]
    term = words[-1] if words else ""
    if term:
        lookup = lookup_term_in_hr_data(term, employees_df, former_df)
        if lookup:
            dept = lookup.get("department_name", term).title()
            headcount = lookup.get("headcount")
            leavers_total = lookup.get("leavers_total")
            leavers_last_year = lookup.get("leavers_last_year")
            payroll = lookup.get("monthly_payroll_qr")
            roles = lookup.get("top_roles")
            term_brk = lookup.get("termination_breakdown")

            lines: List[str] = []
            if headcount is not None:
                lines.append(
                    f"{dept} department is present with about {headcount} employees."
                )
            else:
                lines.append(
                    f"{dept} appears in the HR data, but only partial details are available."
                )

            if payroll is not None:
                lines.append(f"- Estimated monthly payroll: QAR {round(payroll, 2)}")
            if leavers_total is not None:
                lines.append(f"- Total leavers recorded: {leavers_total}")
            if leavers_last_year is not None:
                lines.append(f"- Leavers in the last year: {leavers_last_year}")
            if term_brk:
                lines.append(f"- Termination breakdown: {term_brk}")
            if roles:
                lines.append(f"- Top roles in this department: {roles}")

            return "\n".join(lines)

    # 1) New joiners: count + names (dynamic from new_joiners_df)
    if "new joiner" in q or "new joiners" in q or "recent hires" in q:
        if new_joiners_df.empty:
            return "There is no new joiner data available in the current dataset."

        total_new = len(new_joiners_df)

        # Sirf count wala question
        if "how many" in q or "kitne" in q:
            return f"There are {total_new} new joiners on record."

        # Naam wala question
        if "name" in q or "names" in q or "who" in q:
            if "first_name" in new_joiners_df.columns and "last_name" in new_joiners_df.columns:
                names = [
                    f"{fn} {ln}"
                    for fn, ln in zip(new_joiners_df["first_name"], new_joiners_df["last_name"])
                ]
            else:
                names = new_joiners_df.get("employee_id", []).astype(str).tolist()

            names_display = names[:20]
            names_str = ", ".join(names_display)

            extra = ""
            if total_new > 20:
                extra = f" (showing 20 of {total_new} new joiners)."
            else:
                extra = "."

            return (
                f"There are {total_new} new joiners on record."
                f" Their names are: {names_str}{extra}"
            )

        # Agar generic new joiner question hai
        return f"There are {len(new_joiners_df)} new joiners on record in the current dataset."

    # 2) Nationality questions
    if any(k in q for k in NATIONALITY_KEYWORDS):
        if employees_df.empty or "nationality" not in employees_df.columns:
            return (
                "Nationality data is not available in the current dataset, so I cannot show a breakdown by country."
            )

        nat_series = employees_df["nationality"].str.lower()
        total_emp = len(employees_df)

        # ---------- 1) Generic: names of <nationality> employees ----------
        if ("name" in q or "names" in q or "who" in q):
            target_nat = None
            for nat in nat_series.unique():
                if nat and nat in q:
                    target_nat = nat
                    break

            if "indian" in q:
                target_nat = "indian"
            elif "qatari" in q or "qataris" in q:
                target_nat = "qatari"
            elif "filipino" in q:
                target_nat = "filipino"
            elif "pakistani" in q:
                target_nat = "pakistani"

            if not target_nat:
                return (
                    "Please specify which nationality you want the names for, for example: "
                    "'names of Indian employees' or 'names of Qatari employees'."
                )

            mask = nat_series == target_nat
            sub_df = employees_df[mask]
            count = len(sub_df)
            pct = round((count / total_emp) * 100, 1) if total_emp else 0

            if count == 0:
                return f"There are currently no {target_nat.title()} employees in the workforce."

            if "first_name" in sub_df.columns and "last_name" in sub_df.columns:
                names = [
                    f"{fn} {ln}"
                    for fn, ln in zip(sub_df["first_name"], sub_df["last_name"])
                ]
            else:
                names = sub_df.get("employee_id", []).astype(str).tolist()

            names_str = ", ".join(names)
            return (
                f"We currently have {count} {target_nat.title()} employees, which is about {pct}% of the workforce."
                f" Their names are: {names_str}."
            )

        # ---------- 2) Only count questions (e.g. 'How many Indian employees') ----------
        nat_split = (
            employees_df["nationality"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
        )

        if "indian" in q:
            count = (nat_series == "indian").sum()
            pct = round((count / total_emp) * 100, 1) if total_emp else 0
            return (
                f"We currently have {count} Indian employees, which is about {pct}% of the workforce."
            )

        # Default: overall nationality summary
        if len(nat_split) == 0:
            return "Nationality data exists but has no values in this snapshot."

        top_nat = nat_split.index[0]
        top_pct = nat_split.iloc[0]
        unique_nats = len(nat_split)

        return (
            f"We currently have {unique_nats} distinct nationalities in the workforce."
            f" The largest group is {top_nat}, representing about {top_pct}% of employees."
            f" You can see the full nationality breakdown on the HR dashboard."
        )

    # 3) HR overview + prompt (CEO-style)
    hr_overview = build_hr_overview(
        employees_df=employees_df,
        former_df=former_df,
        base_metrics=base_metrics,
        performance_metrics=performance_metrics,
    )
    employees_metrics = base_metrics or fetch_employee_data()

    q_low = q

    # Workforce health
    if "workforce" in q_low or "overall" in q_low or "how is my workforce" in q_low:
        te = hr_overview.get("total_employees")
        fe = hr_overview.get("former_employees")
        pay = hr_overview.get("monthly_payroll_qr")
        tenure = hr_overview.get("avg_tenure_years")
        nat = hr_overview.get("nationality_split") or {}
        married = hr_overview.get("married_pct")
        single = hr_overview.get("single_pct")
        new_joiners = hr_overview.get("new_joiners_2023_2024")

        top_nat = max(nat, key=nat.get) if nat else "not available"
        top_nat_pct = nat[top_nat] if nat else 0

        return (
            f"You currently have {te} active employees and {fe} former employees."
            f" There are {new_joiners} new joiners on record."
            f" Monthly payroll is about QAR {round(pay, 0)}."
            f" Average tenure is {tenure} years."
            f" The largest nationality group is {top_nat} at around {top_nat_pct}%."
            f" About {married}% of contracts are married and {single}% are single."
        )

    # Turnover cost
    if "turnover cost" in q_low:
        cost = hr_overview.get("turnover_cost_qr")
        tl = hr_overview.get("turnover_last_year")
        brk = hr_overview.get("turnover_breakdown") or {}
        term = brk.get("terminations", 0)
        resign = brk.get("resignations", 0)

        return (
            f"In the last period, {tl} employees left, with {term} terminations and {resign} resignations."
            f" Estimated turnover cost is about QAR {round(cost, 0)}."
            f" You should focus on early leavers and high‑turnover departments to reduce this cost."
        )

    # Performance issues
    if "performance" in q_low or "any performance issues" in q_low:
        poor = hr_overview.get("poor_performers_count")
        exceed = hr_overview.get("exceed_expectations_pct")
        top_perf = hr_overview.get("top_performer_name")
        probation_failed = hr_overview.get("probation_failed_count")

        line1 = (
            f"There are currently {poor} poor performers identified, while about {exceed}% of employees exceed expectations."
        )
        line2 = f" {probation_failed} employees have failed probation." if probation_failed else ""
        line3 = f" The current top performer is {top_perf}." if top_perf else ""
        line4 = " Focus performance discussions and coaching on low performers and high‑risk teams."

        return (line1 + line2 + line3 + line4).strip()

    # Struggling department
    if "which department is struggling" in q_low or "struggling" in q_low:
        dept = hr_overview.get("top_turnover_department")
        pct = hr_overview.get("top_turnover_pct")
        attr = hr_overview.get("attrition_rate")

        return (
            f"The department struggling the most based on turnover is {dept}, with an estimated turnover share of about {pct}% of leavers."
            f" Overall organizational attrition is {attr}%."
            f" You should review workload, leadership, and pay in this department to stabilize it."
        )

    # Fallback → general prompt + LLM
    prompt_text = ceo_prompt.format(
        hr_overview=str(hr_overview),
        employees=str(employees_metrics),
        question=question,
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.1,
        max_tokens=320,
    )

    return resp.choices[0].message.content
