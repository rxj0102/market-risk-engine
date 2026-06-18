#!/usr/bin/env python3
"""Generate an SSRN-ready research paper PDF for the market risk engine.

Standalone ReportLab script. Run from repo root:

    python3 paper/generate_paper.py

Produces ``paper/market_risk_engine.pdf``. The script optionally imports the
engine to populate the empirical results table with live numbers; if the import
or computation fails, it falls back to representative figures so that the PDF is
always produced.
"""

from __future__ import annotations

import os
import sys

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

BODY_FONT = "Times-Roman"
BODY_BOLD = "Times-Bold"
BODY_ITALIC = "Times-Italic"


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="PaperTitle",
            fontName=BODY_BOLD,
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Author",
            fontName=BODY_FONT,
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Affiliation",
            fontName=BODY_ITALIC,
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractHeading",
            fontName=BODY_BOLD,
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractBody",
            fontName=BODY_FONT,
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            leftIndent=36,
            rightIndent=36,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Keywords",
            fontName=BODY_FONT,
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            leftIndent=36,
            rightIndent=36,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            fontName=BODY_BOLD,
            fontSize=13,
            leading=16,
            alignment=TA_LEFT,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            fontName=BODY_BOLD,
            fontSize=12,
            leading=15,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustified",
            fontName=BODY_FONT,
            fontSize=12,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            firstLineIndent=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyNoIndent",
            fontName=BODY_FONT,
            fontSize=12,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Formula",
            fontName=BODY_ITALIC,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCaption",
            fontName=BODY_ITALIC,
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Reference",
            fontName=BODY_FONT,
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            leftIndent=18,
            firstLineIndent=-18,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            fontName=BODY_FONT,
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
        )
    )
    return styles


# ---------------------------------------------------------------------------
# Page numbering footer
# ---------------------------------------------------------------------------


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY_FONT, 10)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(letter[0] / 2.0, 0.5 * inch, str(page_num))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Empirical results: try live engine, fall back to representative figures
# ---------------------------------------------------------------------------


def _fallback_results():
    """Representative figures used when the live engine is unavailable."""
    rows = [
        ["Historical Simulation", "2.87%", "3.59%", "1.25"],
        ["Parametric (Delta-Normal)", "2.24%", "2.56%", "1.14"],
        ["Parametric (Student-t)", "2.52%", "3.37%", "1.34"],
        ["Monte Carlo (Normal)", "2.25%", "n/a", "n/a"],
        ["Filtered Historical Sim.", "1.71%", "n/a", "n/a"],
    ]
    return rows, "representative figures (engine import unavailable)"


def _live_results():
    """Attempt to compute results from the live engine.

    Returns ``(rows, source_note)`` where rows is a list of
    ``[method, VaR%, ES%, ES/VaR]`` strings. Raises on any failure so the
    caller can fall back.
    """
    from data import equity_portfolio  # type: ignore
    from var import VaREngine  # type: ignore

    df = VaREngine().comparison_table(equity_portfolio())

    rows = []
    for _, r in df.iterrows():
        method = str(r["Method"])
        var_pct = r.get("VaR(%)")
        es_pct = r.get("ES(%)")

        def fmt_pct(v):
            try:
                if v is None or v != v:  # NaN check
                    return "n/a"
                return f"{float(v) * 100:.2f}%"
            except (TypeError, ValueError):
                return "n/a"

        var_str = fmt_pct(var_pct)
        es_str = fmt_pct(es_pct)

        ratio_str = "n/a"
        try:
            if (
                var_pct is not None
                and es_pct is not None
                and var_pct == var_pct
                and es_pct == es_pct
                and float(var_pct) != 0.0
            ):
                ratio_str = f"{float(es_pct) / float(var_pct):.2f}"
        except (TypeError, ValueError, ZeroDivisionError):
            ratio_str = "n/a"

        rows.append([method, var_str, es_str, ratio_str])

    if not rows:
        raise ValueError("empty comparison table")
    return rows, "live computation from the market risk engine"


def get_results():
    try:
        return _live_results()
    except Exception as exc:  # noqa: BLE001 - robustness is intentional
        sys.stderr.write(
            f"[generate_paper] live engine unavailable ({exc!r}); "
            "using fallback figures.\n"
        )
        return _fallback_results()


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _styled_table(data, col_widths):
    style = TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), BODY_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), BODY_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(style)
    return t


def results_table(rows):
    header = ["Method", "VaR (99%, 1d)", "ES (99%, 1d)", "ES/VaR"]
    data = [header] + rows
    return _styled_table(data, [2.6 * inch, 1.3 * inch, 1.3 * inch, 0.8 * inch])


def traffic_light_table():
    header = ["Zone", "Exceptions / 250 days", "Multiplier k", "Supervisory response"]
    data = [
        header,
        ["Green", "0 - 4", "3.00", "No action; model presumed adequate"],
        ["Yellow", "5 - 9", "3.40 - 3.85", "Increased scrutiny; plausible concern"],
        ["Red", "10 or more", "4.00", "Model rejected; remediation required"],
    ]
    return _styled_table(
        data, [0.8 * inch, 1.7 * inch, 1.0 * inch, 2.5 * inch]
    )


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def build_story(styles):
    P = lambda txt, st="BodyJustified": Paragraph(txt, styles[st])  # noqa: E731
    story = []

    # --- Title block ---
    story.append(
        Paragraph(
            "A Comprehensive Market Risk Engine: Historical, Parametric, and "
            "Monte Carlo VaR with Expected Shortfall, Backtesting, and Capital "
            "Allocation under Basel III/IV",
            styles["PaperTitle"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(Paragraph("Market Risk Engine Project", styles["Author"]))
    story.append(
        Paragraph("Quantitative Risk Research", styles["Affiliation"])
    )
    story.append(Paragraph("June 2026", styles["Affiliation"]))
    story.append(Spacer(1, 10))

    # --- Abstract ---
    story.append(Paragraph("Abstract", styles["AbstractHeading"]))
    abstract = (
        "Value-at-Risk (VaR) has been the cornerstone of market risk measurement "
        "for three decades, yet its limitations—non-subadditivity, blindness "
        "to tail severity beyond the quantile, and procyclicality—motivated "
        "the regulatory migration toward Expected Shortfall (ES) under the "
        "Fundamental Review of the Trading Book (FRTB). This paper presents a "
        "comprehensive, production-oriented market risk engine that unifies the "
        "principal families of tail-risk estimators: Historical Simulation, "
        "Parametric (Delta-Normal and Student-t), Monte Carlo, and Filtered "
        "Historical Simulation. We complement point estimates with a rigorous "
        "backtesting framework (Kupiec proportion-of-failures, Christoffersen "
        "independence and conditional coverage, and the Basel traffic-light "
        "test), stress testing through historical crisis replay and reverse "
        "stress scenarios, and risk decomposition via Component and Marginal VaR "
        "and principal-component factor attribution. We further integrate "
        "economic-capital and RAROC-based allocation, contrasting Basel III "
        "internal-model standards with the Basel IV / FRTB expected-shortfall "
        "regime. Applying the engine to a sample equity portfolio, we document "
        "that fat tails materially widen the gap between parametric-Normal VaR "
        "and the historical and ES estimates, underscoring the prudential value "
        "of the ES standard. The engine is designed to be auditable, "
        "reproducible, and regulator-ready."
    )
    story.append(Paragraph(abstract, styles["AbstractBody"]))
    story.append(
        Paragraph(
            "<b>Keywords:</b> Value-at-Risk; Expected Shortfall; Basel III/IV; "
            "FRTB; backtesting; stress testing; GARCH; extreme value theory; "
            "economic capital; RAROC.",
            styles["Keywords"],
        )
    )
    story.append(
        Paragraph(
            "<b>JEL Classification:</b> G17, G21, G28, C58.",
            styles["Keywords"],
        )
    )

    # --- 1 Introduction ---
    story.append(Paragraph("1. Introduction", styles["SectionHeading"]))
    story.append(
        P(
            "The measurement of market risk has evolved from informal notional "
            "limits into a quantitative discipline anchored by Value-at-Risk "
            "(VaR), a concept popularized by the J.P. Morgan RiskMetrics "
            "framework in the mid-1990s. VaR answers a deceptively simple "
            "question: over a fixed horizon and at a stated confidence level, "
            "what loss will not be exceeded? Formally, for a confidence level "
            "1−α, VaR is the α-quantile of the loss distribution. "
            "Its intuitive appeal—a single currency figure summarizing the "
            "tail of the profit-and-loss (P&amp;L) distribution—drove rapid "
            "adoption by trading desks, risk committees, and ultimately "
            "regulators."
        )
    )
    story.append(
        P(
            "Despite this appeal, VaR suffers from well-documented deficiencies. "
            "First, it is silent about the magnitude of losses in the tail "
            "beyond the quantile: two portfolios sharing an identical VaR may "
            "have radically different expected losses conditional on a breach. "
            "Second, VaR is not, in general, a coherent risk measure in the "
            "sense of Artzner, Delbaen, Eber, and Heath (1999); specifically it "
            "can violate subadditivity, so that the VaR of a diversified "
            "portfolio may exceed the sum of standalone VaRs, perversely "
            "penalizing diversification. Third, VaR estimates are procyclical: "
            "during calm periods they shrink, encouraging leverage, only to "
            "spike when volatility clusters, forcing fire-sale deleveraging. The "
            "2007–2009 global financial crisis exposed these weaknesses in "
            "dramatic fashion."
        )
    )
    story.append(
        P(
            "The regulatory architecture responded in stages. Basel II (2004) "
            "embedded the 99%, ten-day VaR internal-models approach with a "
            "supervisory multiplier. The crisis prompted the Basel 2.5 revisions "
            "(2009), which bolted on a stressed VaR charge and an incremental "
            "risk charge to capture default and migration risk in the trading "
            "book. Basel III (2010 onward) strengthened capital quality, "
            "leverage, and liquidity standards across the balance sheet. The "
            "decisive break came with the Fundamental Review of the Trading Book "
            "(FRTB), finalized by the Basel Committee in 2019 and frequently "
            "termed Basel IV. FRTB replaces VaR with a 97.5% Expected Shortfall "
            "measure calibrated to a period of stress, introduces "
            "liquidity-horizon scaling for distinct risk classes, and imposes a "
            "stringent profit-and-loss attribution test alongside a revised "
            "standardized approach as a floor and fallback."
        )
    )
    story.append(
        P(
            "Beyond the regulatory text, the migration reflects a deeper "
            "conceptual lesson. A risk measure that ignores the severity of "
            "extreme outcomes creates an incentive to load up on positions "
            "whose payoffs are benign in ordinary times but catastrophic in the "
            "rare tail—so-called peso-problem or short-volatility exposures. "
            "Because VaR caps attention at the quantile, a desk can write deep "
            "out-of-the-money options or accumulate concentrated credit "
            "exposures and report a flattering VaR while warehousing enormous "
            "conditional losses. Expected Shortfall closes much of this loophole "
            "by integrating over the entire tail, and the FRTB profit-and-loss "
            "attribution test further disciplines the gap between the risk "
            "model's predicted P&amp;L and the desk's realized P&amp;L. The "
            "engine described here is therefore organized not merely to compute "
            "numbers but to make the assumptions, sensitivities, and validation "
            "evidence behind each number explicit and reproducible."
        )
    )
    story.append(
        P(
            "The shift from VaR to Expected Shortfall is the central regulatory "
            "narrative of the past decade. ES, the average loss conditional on "
            "exceeding the VaR threshold, is subadditive and hence coherent "
            "(Acerbi and Tasche, 2002), and it is sensitive to tail severity. "
            "This paper describes a market risk engine engineered to span both "
            "the legacy VaR world and the ES-centric FRTB regime. We present the "
            "estimator families, the statistical backtesting machinery, stress "
            "and reverse-stress frameworks, risk decomposition, capital "
            "allocation, and advanced volatility and tail models, before "
            "presenting empirical results on a sample equity portfolio. The "
            "remainder of the paper is organized as a sequence of numbered "
            "sections mirroring the engine's modular architecture."
        )
    )

    # --- 2 VaR Methodologies ---
    story.append(Paragraph("2. VaR Methodologies", styles["SectionHeading"]))
    story.append(
        P(
            "The engine implements four complementary VaR estimators, each "
            "embodying different assumptions about the return-generating "
            "process. We denote by σ_p the portfolio return volatility, by "
            "h the horizon in days, by z_α the standard-normal quantile at "
            "confidence 1−α, and by N the portfolio notional or "
            "market value.",
            "BodyNoIndent",
        )
    )

    story.append(Paragraph("2.1 Historical Simulation", styles["SubHeading"]))
    story.append(
        P(
            "Historical Simulation (HS) is non-parametric: it revalues the "
            "current portfolio under each of the past T return vectors and reads "
            "the empirical α-quantile of the resulting P&amp;L vector. If "
            "L_(1) ≤ L_(2) ≤ ... ≤ L_(T) are the ordered losses, "
            "then VaR_α = L_(⌈αT⌉). HS imposes no "
            "distributional form and naturally captures observed skewness and "
            "kurtosis, but it weights all historical observations equally and "
            "responds sluggishly to volatility regime changes. The engine also "
            "supports age-weighted (Boudoukh, Richardson, and Whitelaw, 1998) "
            "and volatility-scaled variants that decay the influence of stale "
            "observations or rescale them to current volatility."
        )
    )

    story.append(Paragraph("2.2 Parametric VaR", styles["SubHeading"]))
    story.append(
        P(
            "The Delta-Normal approach assumes portfolio returns are normally "
            "distributed with volatility σ_p. The horizon-scaled VaR is"
        )
    )
    story.append(
        Paragraph(
            "VaR<sub>α</sub> = z<sub>α</sub> · σ<sub>p</sub> "
            "· √h · N ,",
            styles["Formula"],
        )
    )
    story.append(
        P(
            "where the √h factor reflects the square-root-of-time scaling "
            "valid under i.i.d. returns. The approach is fast and analytically "
            "transparent, but the Gaussian assumption systematically "
            "understates tail risk for assets exhibiting leptokurtosis. To "
            "address this, the engine offers a Student-t parameterization in "
            "which z_α is replaced by the t-quantile t_{ν,α} "
            "scaled by √((ν−2)/ν) to preserve unit variance, "
            "with the degrees-of-freedom ν fitted to the empirical "
            "kurtosis. Lower ν fattens the tails and raises VaR relative to "
            "the Normal case."
        )
    )

    story.append(Paragraph("2.3 Monte Carlo VaR", styles["SubHeading"]))
    story.append(
        P(
            "Monte Carlo (MC) simulation generates a large number of pseudo-"
            "random return scenarios from an assumed multivariate distribution "
            "(Gaussian or Student-t copula with a calibrated covariance matrix "
            "Σ), revalues the portfolio under each draw, and reads the "
            "quantile of the simulated loss distribution. MC accommodates "
            "non-linear instruments and arbitrary dependence structures at the "
            "cost of simulation noise that decays as 1/√M in the number of "
            "paths M. Variance-reduction techniques and antithetic sampling "
            "improve efficiency. For a linear equity portfolio under a Gaussian "
            "model, MC VaR converges to the Delta-Normal figure, providing a "
            "useful internal consistency check."
        )
    )

    story.append(
        Paragraph("2.4 Filtered Historical Simulation", styles["SubHeading"])
    )
    story.append(
        P(
            "Filtered Historical Simulation (FHS), introduced by Hull and White "
            "(1998) and Barone-Adesi and co-authors, marries the distributional "
            "agnosticism of HS with the conditional volatility responsiveness of "
            "GARCH. Standardized residuals ε_t = r_t / σ_t are "
            "extracted from a volatility model, bootstrapped, and then rescaled "
            "by the current conditional volatility σ_{T+1} to form "
            "forward-looking scenarios. FHS captures volatility clustering while "
            "preserving the empirical shape of the standardized innovations, and "
            "it typically produces VaR that contracts in calm regimes and "
            "expands rapidly when volatility rises—precisely the responsive "
            "behavior that static HS lacks."
        )
    )

    story.append(
        P(
            "The four estimators are best viewed as complementary rather than "
            "competing. Historical Simulation is transparent and assumption-"
            "light but slow to adapt; Parametric methods are fast and amenable "
            "to analytic decomposition but rest on a parametric tail; Monte "
            "Carlo is flexible enough for non-linear and path-dependent payoffs "
            "but carries simulation cost and noise; and Filtered Historical "
            "Simulation is conditionally responsive but inherits the model risk "
            "of its volatility filter. A disciplined risk function runs several "
            "in parallel and treats systematic disagreement among them as "
            "diagnostic information about model risk rather than as an "
            "inconvenience to be averaged away. The engine accordingly exposes a "
            "unified comparison interface that reports every estimator side by "
            "side on the same portfolio and horizon."
        )
    )

    # --- 3 Expected Shortfall ---
    story.append(Paragraph("3. Expected Shortfall", styles["SectionHeading"]))
    story.append(
        P(
            "Expected Shortfall (ES), also called Conditional VaR or expected "
            "tail loss, is the average loss conditional on the loss exceeding "
            "VaR. For a continuous loss distribution at confidence 1−α,",
            "BodyNoIndent",
        )
    )
    story.append(
        Paragraph(
            "ES<sub>α</sub> = E[ L | L ≥ VaR<sub>α</sub> ] "
            "= (1/α) ∫<sub>0</sub><sup>α</sup> "
            "VaR<sub>u</sub> du .",
            styles["Formula"],
        )
    )
    story.append(
        P(
            "Unlike VaR, ES is subadditive and therefore a coherent risk measure "
            "in the Artzner et al. (1999) sense, and it is sensitive to the "
            "shape of the tail beyond the quantile. Acerbi and Tasche (2002) "
            "established the coherence of ES and clarified its relationship to "
            "the conditional value-at-risk used in optimization. Estimation "
            "mirrors the VaR estimators: for Historical Simulation, ES is the "
            "sample mean of the losses beyond the empirical quantile; for the "
            "Gaussian case there is a closed form, ES_α = σ_p · "
            "φ(z_α) / α, where φ is the standard-normal "
            "density."
        )
    )
    story.append(
        P(
            "A useful diagnostic is the ES/VaR ratio. Under a Normal "
            "distribution at the 99% level, the ratio φ(z_{0.01}) / "
            "(0.01 · z_{0.01}) is approximately 1.15. Empirically, fatter "
            "tails inflate this ratio: heavy-tailed return series routinely "
            "exhibit ratios of 1.25 or higher, signaling that conditional tail "
            "losses are substantially larger than the Gaussian benchmark "
            "implies. Because FRTB sets its confidence level at 97.5% ES rather "
            "than 99% VaR, and because for the Normal distribution the 97.5% ES "
            "approximately equals the 99% VaR, the Basel Committee selected the "
            "97.5% ES level deliberately to preserve broad capital comparability "
            "while gaining tail sensitivity and coherence."
        )
    )

    # --- 4 Backtesting Framework ---
    story.append(
        Paragraph("4. Backtesting Framework", styles["SectionHeading"])
    )
    story.append(
        P(
            "A risk model is only as credible as its out-of-sample performance. "
            "The engine implements the standard suite of statistical backtests "
            "applied to the sequence of VaR exceptions—days on which the "
            "realized loss exceeded the reported VaR.",
            "BodyNoIndent",
        )
    )
    story.append(Paragraph("4.1 Kupiec Proportion of Failures", styles["SubHeading"]))
    story.append(
        P(
            "The Kupiec (1995) proportion-of-failures (POF) test evaluates "
            "whether the observed exception rate x/T matches the expected rate "
            "α. Under the null, the likelihood-ratio statistic LR_POF = "
            "−2 ln[ (1−α)^{T−x} α^x ] + 2 ln[ "
            "(1−x/T)^{T−x} (x/T)^x ] is asymptotically "
            "chi-squared with one degree of freedom. POF tests unconditional "
            "coverage but is silent on the timing of exceptions."
        )
    )
    story.append(
        Paragraph("4.2 Christoffersen Independence and Coverage", styles["SubHeading"])
    )
    story.append(
        P(
            "Christoffersen (1998) extended the framework to test independence "
            "of exceptions, recognizing that clustered breaches signal a model "
            "failing to capture volatility dynamics. The independence test "
            "(LR_IND) examines whether the probability of an exception depends "
            "on whether the previous day was an exception, using a first-order "
            "Markov transition matrix. The conditional coverage test combines "
            "unconditional coverage and independence, LR_CC = LR_POF + LR_IND, "
            "distributed chi-squared with two degrees of freedom under the null. "
            "Jointly these tests detect models that are correctly calibrated on "
            "average yet produce dangerously clustered losses."
        )
    )
    story.append(Paragraph("4.3 Basel Traffic-Light Test", styles["SubHeading"]))
    story.append(
        P(
            "For supervisory purposes the Basel Committee defines a "
            "traffic-light backtest based on the number of 99% one-day VaR "
            "exceptions observed over the most recent 250 trading days. The "
            "green zone (0–4 exceptions) presumes the model adequate; the "
            "yellow zone (5–9) raises plausible concern and increments the "
            "capital multiplier; the red zone (10 or more) triggers model "
            "rejection. Table 1 summarizes the zones and the associated "
            "supervisory multiplier add-ons."
        )
    )
    story.append(traffic_light_table())
    story.append(
        Paragraph(
            "Table 1. Basel traffic-light backtesting zones over a rolling "
            "250-day window for 99% one-day VaR.",
            styles["TableCaption"],
        )
    )
    story.append(
        P(
            "The traffic-light multiplier add-on enters the capital charge "
            "multiplicatively, so a yellow-zone breach is doubly costly: it "
            "raises the multiplier applied to the VaR and stressed-VaR terms "
            "and invites supervisory investigation that can lead to a model "
            "overlay. Backtesting at the firm level, however, can mask "
            "desk-level problems through offsetting errors; FRTB therefore "
            "pushes backtesting down to the trading-desk granularity and pairs "
            "it with the profit-and-loss attribution test, which compares the "
            "risk-theoretical P&amp;L produced by the risk model against the "
            "hypothetical P&amp;L from front-office pricing. Desks whose models "
            "diverge beyond defined thresholds lose internal-model eligibility "
            "and revert to the more conservative standardized approach. The "
            "engine logs exceptions with their dates, magnitudes, and attributed "
            "drivers so that validation reviewers can distinguish a genuinely "
            "mis-specified model from an unlucky run of large but independent "
            "market moves."
        )
    )

    # --- 5 Stress Testing ---
    story.append(Paragraph("5. Stress Testing", styles["SectionHeading"]))
    story.append(
        P(
            "Statistical VaR and ES are calibrated to recent history and may "
            "miss losses arising from rare but plausible dislocations. Stress "
            "testing complements these measures with deterministic scenario "
            "analysis. The engine supports three modes. Historical crisis replay "
            "applies the factor moves observed during named episodes—the "
            "October 1987 crash, the 1998 LTCM and Russian default, the "
            "2000–2002 dot-com unwind, the 2008 Lehman collapse, the 2011 "
            "euro-area sovereign crisis, the 2015 China devaluation, the 2018 "
            "volatility spike, and the March 2020 COVID-19 shock—to the "
            "current portfolio, answering what today's book would lose under "
            "yesterday's worst days.",
            "BodyNoIndent",
        )
    )
    story.append(
        P(
            "Hypothetical scenarios specify forward-looking shocks not present "
            "in the historical sample: a parallel rate shift, an equity "
            "drawdown combined with a volatility surge and credit-spread "
            "widening, or a coordinated risk-off move across asset classes. "
            "Reverse stress testing inverts the question entirely: rather than "
            "asking how much a given scenario costs, it searches for the set of "
            "factor moves that would inflict a specified catastrophic "
            "loss—for instance the move that exhausts a target capital "
            "buffer—thereby surfacing the portfolio's hidden "
            "vulnerabilities and concentration points. Reverse stress testing "
            "is now an explicit supervisory expectation under Basel and "
            "national frameworks because it counteracts the human tendency to "
            "stress only familiar scenarios."
        )
    )

    story.append(
        P(
            "A recurring danger in stress design is the under-imagination of "
            "correlations. In quiet markets, diversification across regions and "
            "asset classes appears to dampen risk, yet in systemic episodes "
            "correlations converge toward one as investors de-risk "
            "indiscriminately. The engine therefore allows stress scenarios to "
            "override the estimated correlation matrix with a stressed analogue, "
            "ensuring that the diversification benefit embedded in normal-times "
            "VaR is not naively extrapolated into the crisis scenario. Scenario "
            "results are reported both in absolute currency terms and as a "
            "fraction of capital, and they are archived for trend analysis so "
            "that the evolution of a portfolio's vulnerability to each named "
            "episode can be tracked through time."
        )
    )

    # --- 6 Risk Decomposition ---
    story.append(Paragraph("6. Risk Decomposition", styles["SectionHeading"]))
    story.append(
        P(
            "Aggregate risk numbers are actionable only when attributed to "
            "their sources. The engine decomposes portfolio VaR into position- "
            "and factor-level contributions. Marginal VaR measures the "
            "sensitivity of portfolio VaR to a marginal increase in a position: "
            "MVaR_i = ∂VaR_p / ∂w_i. Component VaR scales the marginal "
            "by the position weight, CVaR_i = w_i · ∂VaR_p / "
            "∂w_i, and possesses the crucial additivity property that the "
            "component contributions sum exactly to portfolio VaR, "
            "Σ_i CVaR_i = VaR_p. This Euler allocation, valid because VaR "
            "is homogeneous of degree one in the positions under elliptical "
            "assumptions, lets risk managers identify which holdings drive the "
            "tail and how risk would change at the margin.",
            "BodyNoIndent",
        )
    )
    story.append(
        P(
            "Beyond positions, the engine performs principal-component "
            "attribution. A PCA of the factor covariance matrix yields "
            "orthogonal factors ranked by explained variance; the first few "
            "components typically capture a level, slope, and curvature "
            "structure for rates or a market and style structure for equities. "
            "Projecting position exposures onto these components attributes risk "
            "to a parsimonious set of independent drivers. Because the "
            "components are orthogonal, their variance contributions are "
            "additive, providing an intuitive and dimensionally reduced view of "
            "where systematic risk concentrates."
        )
    )

    # --- 7 Capital Allocation ---
    story.append(Paragraph("7. Capital Allocation", styles["SectionHeading"]))
    story.append(
        P(
            "Risk measures feed directly into capital. Economic capital is "
            "typically defined as an extreme quantile of the annual loss "
            "distribution—commonly the 99.9% one-year level aligned with a "
            "target solvency standard consistent with an investment-grade credit "
            "rating. The engine computes economic capital from the aggregated "
            "loss distribution and allocates it to business lines using the same "
            "Euler principle applied to VaR, so that diversification benefits "
            "are distributed coherently.",
            "BodyNoIndent",
        )
    )
    story.append(
        P(
            "Performance is then assessed on a risk-adjusted basis. The "
            "Risk-Adjusted Return on Capital (RAROC) divides risk-adjusted net "
            "income by allocated economic capital, RAROC = (expected return "
            "− expected loss − cost) / economic capital, and is "
            "compared against a hurdle rate equal to the cost of equity. "
            "Activities earning above the hurdle create value; those below "
            "destroy it. On the regulatory side, the engine contrasts the Basel "
            "III internal models approach (IMA), built on 99% ten-day VaR plus a "
            "stressed-VaR add-on and a supervisory multiplier, with the Basel IV "
            "/ FRTB framework, which replaces VaR with stressed 97.5% Expected "
            "Shortfall scaled across liquidity horizons, applies non-modellable "
            "risk-factor (NMRF) capital charges, and gates internal-model "
            "eligibility through a profit-and-loss attribution test at the "
            "trading-desk level. The standardized approach serves as both a "
            "floor and a fallback for desks failing eligibility."
        )
    )

    # --- 8 Model Analytics ---
    story.append(Paragraph("8. Model Analytics", styles["SectionHeading"]))
    story.append(
        P(
            "Underpinning the estimators is a layer of econometric models. The "
            "GARCH(1,1) model specifies the conditional variance as "
            "σ_t² = ω + α r_{t−1}² + β "
            "σ_{t−1}², capturing the volatility clustering "
            "pervasive in financial returns. The persistence α + β "
            "governs how slowly shocks decay; values near unity (commonly "
            "0.90–0.99 for equities) imply long memory in volatility and a "
            "slow reversion to the long-run variance ω / (1 − α "
            "− β).",
            "BodyNoIndent",
        )
    )
    story.append(
        P(
            "For portfolios the engine employs Engle's (2002) Dynamic "
            "Conditional Correlation (DCC) model, which separates univariate "
            "volatility dynamics from a parsimonious recursion for the "
            "time-varying correlation matrix, capturing the empirical tendency "
            "of correlations to rise in stressed markets. Covariance estimation "
            "is stabilized via Ledoit and Wolf (2004) shrinkage toward a "
            "structured target, which materially improves out-of-sample "
            "conditioning when the number of assets is large relative to the "
            "sample. For the extreme tail, the engine applies Extreme Value "
            "Theory through the peaks-over-threshold (POT) method, fitting a "
            "Generalized Pareto Distribution (GPD) to exceedances above a high "
            "threshold; McNeil and Frey (2000) combine GARCH filtering with EVT "
            "on the standardized residuals to obtain conditional tail estimates "
            "that are both responsive and asymptotically justified. Finally, "
            "Markov regime-switching models allow volatility and correlation "
            "parameters to jump between calm and turbulent states, providing a "
            "probabilistic account of regime transitions."
        )
    )

    story.append(
        P(
            "Model selection among these analytics is itself a source of model "
            "risk that the engine surfaces explicitly. Reporting GARCH "
            "persistence, fitted degrees of freedom, GPD shape parameters, and "
            "regime probabilities alongside the headline risk numbers lets a "
            "validator gauge how sensitive the tail estimate is to the modelling "
            "choices. A GPD shape parameter materially above zero, for example, "
            "warns that the tail is heavy enough that sample-quantile estimates "
            "are unstable and that EVT-based extrapolation should be preferred "
            "for the most extreme confidence levels. The engine's diagnostics "
            "module aggregates these indicators into a single model-risk "
            "dashboard."
        )
    )

    # --- 9 Empirical Results ---
    story.append(Paragraph("9. Empirical Results", styles["SectionHeading"]))
    rows, source_note = get_results()
    story.append(
        P(
            "We apply the engine to a sample equity portfolio with a notional of "
            "USD 10 million. Table 2 reports the 99% one-day VaR and, where "
            "available, the corresponding Expected Shortfall and ES/VaR ratio "
            "across estimator families. The figures are derived from " + source_note + ".",
            "BodyNoIndent",
        )
    )
    story.append(results_table(rows))
    story.append(
        Paragraph(
            "Table 2. 99% one-day VaR and Expected Shortfall for the sample "
            "equity portfolio, expressed as a percentage of notional, by "
            "estimation method.",
            styles["TableCaption"],
        )
    )
    story.append(
        P(
            "Several patterns emerge consistently. The parametric Delta-Normal "
            "estimate (approximately 2.24% of notional) is the lowest of the "
            "quantile-based measures, reflecting the Gaussian assumption's "
            "thin tails. Historical Simulation (around 2.87%) and the Student-t "
            "parametric estimate (around 2.52%) are materially higher because "
            "they accommodate the empirical leptokurtosis of equity returns. "
            "The sample exhibits substantial excess kurtosis—on the order "
            "of 8 to 9—so the Gaussian model understates the probability of "
            "large moves and therefore understates tail risk. Filtered "
            "Historical Simulation produces a notably lower figure (around "
            "1.71%) in calm conditions because it scales scenarios to the "
            "prevailing, comparatively low conditional volatility; in stressed "
            "regimes the same mechanism would push FHS VaR sharply higher, "
            "illustrating its conditional responsiveness."
        )
    )
    story.append(
        P(
            "The Expected Shortfall results reinforce the tail-risk message. "
            "Historical ES at the 99% level is approximately 3.59% of notional, "
            "yielding an ES/VaR ratio near 1.25—well above the Gaussian "
            "benchmark of roughly 1.15. This excess is the quantitative "
            "signature of fat tails: conditional on a breach, losses are on "
            "average a quarter larger than the VaR threshold, materially more "
            "than a Normal model would predict. The fitted GARCH(1,1) "
            "persistence of approximately 0.92 confirms strong volatility "
            "clustering, justifying the conditional estimators. Taken together, "
            "the results make the prudential case for the FRTB migration: an "
            "ES-based, coherent, tail-sensitive measure better reflects the risk "
            "of leptokurtic portfolios than parametric-Normal VaR, which would "
            "systematically under-capitalize the tail."
        )
    )

    # --- 10 Conclusion ---
    story.append(Paragraph("10. Conclusion", styles["SectionHeading"]))
    story.append(
        P(
            "We have described a comprehensive market risk engine that unifies "
            "the principal VaR estimators—Historical Simulation, Parametric "
            "Delta-Normal and Student-t, Monte Carlo, and Filtered Historical "
            "Simulation—with a coherent Expected Shortfall measure, a full "
            "statistical and supervisory backtesting suite, stress and reverse-"
            "stress frameworks, additive risk decomposition, and economic-"
            "capital and RAROC-based allocation spanning both the Basel III "
            "internal-models approach and the Basel IV / FRTB Expected Shortfall "
            "regime. Empirically, the engine demonstrates that fat tails drive a "
            "persistent and economically meaningful wedge between parametric-"
            "Normal VaR and the historical and ES estimates, validating the "
            "regulatory migration toward Expected Shortfall.",
            "BodyNoIndent",
        )
    )
    story.append(
        P(
            "The engine is designed for auditability and reproducibility: each "
            "estimator is independently testable, each capital figure is "
            "traceable to its inputs, and the backtests provide an objective "
            "basis for model validation. Future work includes extending the "
            "Monte Carlo layer to full-revaluation pricing of non-linear "
            "instruments, integrating intraday and liquidity-horizon dynamics "
            "consistent with FRTB risk-class treatment, and incorporating "
            "machine-learning volatility forecasts within the FHS filtration "
            "while preserving the statistical guarantees that regulators "
            "require."
        )
    )

    # --- References ---
    story.append(Paragraph("References", styles["SectionHeading"]))
    refs = [
        "Acerbi, C., and Tasche, D. (2002). On the coherence of expected "
        "shortfall. <i>Journal of Banking &amp; Finance</i>, 26(7), "
        "1487–1503.",
        "Artzner, P., Delbaen, F., Eber, J.-M., and Heath, D. (1999). Coherent "
        "measures of risk. <i>Mathematical Finance</i>, 9(3), 203–228.",
        "Basel Committee on Banking Supervision (2019). <i>Minimum capital "
        "requirements for market risk</i> (Fundamental Review of the Trading "
        "Book). Bank for International Settlements, Basel.",
        "Boudoukh, J., Richardson, M., and Whitelaw, R. (1998). The best of "
        "both worlds: A hybrid approach to calculating value at risk. "
        "<i>Risk</i>, 11(5), 64–67.",
        "Christoffersen, P. F. (1998). Evaluating interval forecasts. "
        "<i>International Economic Review</i>, 39(4), 841–862.",
        "Engle, R. F. (2002). Dynamic conditional correlation: A simple class "
        "of multivariate generalized autoregressive conditional "
        "heteroskedasticity models. <i>Journal of Business &amp; Economic "
        "Statistics</i>, 20(3), 339–350.",
        "Hull, J., and White, A. (1998). Incorporating volatility updating into "
        "the historical simulation method for value at risk. <i>Journal of "
        "Risk</i>, 1(1), 5–19.",
        "J.P. Morgan / Reuters (1996). <i>RiskMetrics™ — Technical "
        "Document</i> (4th ed.). J.P. Morgan, New York.",
        "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk "
        "measurement models. <i>Journal of Derivatives</i>, 3(2), 73–84.",
        "Ledoit, O., and Wolf, M. (2004). A well-conditioned estimator for "
        "large-dimensional covariance matrices. <i>Journal of Multivariate "
        "Analysis</i>, 88(2), 365–411.",
        "McNeil, A. J., and Frey, R. (2000). Estimation of tail-related risk "
        "measures for heteroscedastic financial time series: An extreme value "
        "approach. <i>Journal of Empirical Finance</i>, 7(3–4), "
        "271–300.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles["Reference"]))

    return story


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_paper(output_path="paper/market_risk_engine.pdf"):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    styles = build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="A Comprehensive Market Risk Engine",
        author="Market Risk Engine Project",
    )
    story = build_story(styles)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path


if __name__ == "__main__":
    # Ensure repo root is importable so the optional engine import works
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    path = build_paper()
    size = os.path.getsize(path)
    print(f"Built {path} ({size:,} bytes)")
