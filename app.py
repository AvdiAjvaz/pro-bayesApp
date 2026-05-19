"""Aplikacion Streamlit per Probabilitetin Total dhe Formulen e Bayes-it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from probability_solver import ProbabilitySolver, ScenarioItem


sns.set_theme(style="whitegrid")


def percent(value: float) -> str:
    return ProbabilitySolver.format_percent(value)


def make_solver_from_percent(items: list[tuple[str, float, float]]) -> ProbabilitySolver:
    return ProbabilitySolver(
        [
            ScenarioItem(
                name=name,
                prior=ProbabilitySolver.percent_to_probability(prior_percent),
                likelihood=ProbabilitySolver.percent_to_probability(likelihood_percent),
            )
            for name, prior_percent, likelihood_percent in items
        ]
    )


def show_prior_warning(solver: ProbabilitySolver, label: str) -> None:
    prior_sum = solver.prior_sum()
    if abs(prior_sum - 1) > 0.0001:
        st.warning(
            f"Shuma e probabiliteteve fillestare per {label} eshte {percent(prior_sum)}, "
            "jo 100.00%. Rezultatet llogariten me vlerat aktuale."
        )


def contributions_dataframe(solver: ProbabilitySolver) -> pd.DataFrame:
    df = pd.DataFrame(solver.contributions())
    for column in ["P(Bi)", "P(E|Bi)", "Kontributi"]:
        df[column] = df[column].map(percent)
    return df


def plot_contributions(solver: ProbabilitySolver, title: str, x_label: str, y_label: str):
    raw_df = pd.DataFrame(solver.contributions())
    raw_df["Kontributi (%)"] = raw_df["Kontributi"] * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=raw_df, x="Emri", y="Kontributi (%)", hue="Emri", palette="Set2", ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3)
    fig.tight_layout()
    return fig


def plot_prior_vs_updated(prior: float, updated: float):
    df = pd.DataFrame(
        {
            "Lloji": ["Probabiliteti fillestar P(S)", "Pas testit pozitiv P(S|+)"],
            "Probabiliteti (%)": [prior * 100, updated * 100],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="Lloji", y="Probabiliteti (%)", hue="Lloji", palette="Set1", ax=ax, legend=False)
    ax.set_title("Probabiliteti fillestar kundrejt probabilitetit te perditesuar")
    ax.set_xlabel("")
    ax.set_ylabel("Probabiliteti (%)")
    ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3)
    ax.tick_params(axis="x", rotation=8)
    fig.tight_layout()
    return fig


def render_factory_example() -> None:
    st.header("1. Shembulli i Fabrikes")
    st.write("Llogaritja e probabilitetit total te defektit dhe probabilitetit qe defekti vjen nga Linja 3.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prodhimi")
        b1 = st.number_input("Linja 1 - P(B1) %", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        b2 = st.number_input("Linja 2 - P(B2) %", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        b3 = st.number_input("Linja 3 - P(B3) %", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with col2:
        st.subheader("Shkalla e defekteve")
        d_b1 = st.number_input("P(D|B1) %", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
        d_b2 = st.number_input("P(D|B2) %", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
        d_b3 = st.number_input("P(D|B3) %", min_value=0.0, max_value=100.0, value=6.0, step=0.1)

    solver = make_solver_from_percent(
        [
            ("B1", b1, d_b1),
            ("B2", b2, d_b2),
            ("B3", b3, d_b3),
        ]
    )
    show_prior_warning(solver, "linjat e prodhimit")

    total_defect = solver.total_probability()
    st.subheader("Hapat e llogaritjes")
    st.latex(r"P(D)=P(B_1)P(D|B_1)+P(B_2)P(D|B_2)+P(B_3)P(D|B_3)")
    st.code(solver.total_formula_steps("D"), language="text")
    st.success(f"Probabiliteti total i defektit P(D) = {percent(total_defect)}")

    if total_defect == 0:
        st.error("P(D) eshte 0, prandaj P(B3|D) nuk mund te llogaritet me Bayes.")
    else:
        b3_given_d = solver.bayes_probability("B3")
        st.latex(r"P(B_3|D)=\frac{P(B_3)P(D|B_3)}{P(D)}")
        st.code(solver.bayes_formula_steps("B3", "D"), language="text")
        st.info(f"Nese produkti eshte me defekt, probabiliteti qe ka ardhur nga Linja 3 eshte {percent(b3_given_d)}.")

    st.subheader("Tabela e kontributeve")
    st.dataframe(contributions_dataframe(solver), use_container_width=True)
    st.pyplot(plot_contributions(solver, "Kontributi i linjave ne defektin total", "Linja", "Kontributi ne P(D)"))


def render_medical_example() -> None:
    st.header("2. Shembulli Mjekesor")
    st.write("Llogaritja e probabilitetit total qe testi del pozitiv dhe probabilitetit real te semundjes pas testit pozitiv.")

    col1, col2 = st.columns(2)
    with col1:
        sick = st.number_input("Sëmundja - P(S) %", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
        healthy = st.number_input("Të shëndetshëm - P(H) %", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
    with col2:
        positive_sick = st.number_input("Test pozitiv nëse je i sëmurë - P(+|S) %", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        positive_healthy = st.number_input("False Positive - P(+|H) %", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

    solver = make_solver_from_percent(
        [
            ("S", sick, positive_sick),
            ("H", healthy, positive_healthy),
        ]
    )
    show_prior_warning(solver, "gjendjet S dhe H")

    total_positive = solver.total_probability()
    st.subheader("Hapat e llogaritjes")
    st.latex(r"P(+)=P(S)P(+|S)+P(H)P(+|H)")
    st.code(solver.total_formula_steps("+"), language="text")
    st.success(f"Probabiliteti total qe testi te dale pozitiv P(+) = {percent(total_positive)}")

    if total_positive == 0:
        st.error("P(+) eshte 0, prandaj P(S|+) nuk mund te llogaritet me Bayes.")
    else:
        sick_given_positive = solver.bayes_probability("S")
        st.latex(r"P(S|+)=\frac{P(S)P(+|S)}{P(+)}")
        st.code(solver.bayes_formula_steps("S", "+"), language="text")
        st.info(f"Nese testi del pozitiv, probabiliteti real qe personi eshte i semure eshte {percent(sick_given_positive)}.")
        st.pyplot(
            plot_prior_vs_updated(
                ProbabilitySolver.percent_to_probability(sick),
                sick_given_positive,
            )
        )

    st.subheader("Tabela e kontributeve")
    st.dataframe(contributions_dataframe(solver), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Probabiliteti Total dhe Bayes",
        page_icon="📊",
        layout="wide",
    )

    st.title("Probabiliteti Total dhe Formula e Bayes-it")
    st.write(
        "Aplikacion interaktiv per dy shembuj praktike: prodhimi ne fabrike dhe diagnostikimi mjekesor."
    )

    tab_factory, tab_medical = st.tabs(["Fabrika", "Diagnostikimi Mjekesor"])
    with tab_factory:
        render_factory_example()
    with tab_medical:
        render_medical_example()


if __name__ == "__main__":
    main()
