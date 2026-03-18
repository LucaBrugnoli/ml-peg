"""Build main Dash application."""

from __future__ import annotations

from importlib import import_module
import warnings

from dash import Dash, Input, Output, callback
from dash.dash_table import DataTable
from dash.dcc import Loading, Store, Tab, Tabs
from dash.html import H1, H3, Div, Img, Span
from yaml import safe_load

from ml_peg.analysis.utils.utils import calc_table_scores, get_table_style
from ml_peg.app import APP_ROOT
from ml_peg.app.utils.build_components import (
    build_faqs,
    build_footer,
    build_weight_components,
)
from ml_peg.app.utils.onboarding import (
    build_onboarding_modal,
    build_tutorial_button,
    register_onboarding_callbacks,
)
from ml_peg.app.utils.register_callbacks import register_benchmark_to_category_callback
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
    calculate_column_widths,
    get_framework_config,
    load_model_registry_configs,
    sig_fig_format,
)
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)


def get_all_tests(
    category: str = "*",
) -> tuple[
    dict[str, dict[str, list[Div]]],
    dict[str, dict[str, DataTable]],
    dict[str, dict[str, str]],
]:
    """
    Get layout and register callbacks for all categories.

    Parameters
    ----------
    category
        Name of category directory to search for tests. Default is '*'.

    Returns
    -------
    tuple
        Layouts, tables, and framework IDs for all categories.
    """
    # Find Python files e.g. app_OC157.py in mlip_tesing.app module.
    # We will get the category from the parent's parent directory
    # E.g. ml_peg/app/surfaces/OC157/app_OC157.py -> surfaces
    tests = APP_ROOT.glob(f"{category}/*/app*.py")
    layouts = {}
    tables = {}
    frameworks = {}

    # Build all layouts, and register all callbacks to main app.
    for test in tests:
        try:
            # Import test layout/callbacks
            test_name = test.parent.name
            category_name = test.parent.parent.name
            test_module = import_module(
                f"ml_peg.app.{category_name}.{test_name}.app_{test_name}"
            )
            test_app = test_module.get_app()

            # Get layouts and tables for each category/test
            if category_name not in layouts:
                layouts[category_name] = {}
                tables[category_name] = {}
                frameworks[category_name] = {}
            layouts[category_name][test_app.name] = test_app.layout
            tables[category_name][test_app.name] = test_app.table
            frameworks[category_name][test_app.name] = test_app.framework_id
        except FileNotFoundError as err:
            warnings.warn(
                f"Unable to load layout for {test_name} in {category_name} category. "
                f"Full error:\n{err}",
                stacklevel=2,
            )
            continue

        # Register test callbacks
        try:
            test_app.register_callbacks()
        except FileNotFoundError as err:
            warnings.warn(
                f"Unable to register callbacks for {test_name} in {category_name} "
                f"category. Full error:\n{err}",
                stacklevel=2,
            )
            continue

    return layouts, tables, frameworks


def build_category(
    all_layouts: dict[str, dict[str, list[Div]]],
    all_tables: dict[str, dict[str, DataTable]],
    all_frameworks: dict[str, dict[str, str]],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, DataTable],
    dict[str, float],
    set[str],
]:
    """
    Build category layouts and summary tables.

    Parameters
    ----------
    all_layouts
        Layouts of all tests, grouped by category.
    all_tables
        Tables for all tests, grouped by category.
    all_frameworks
        Framework IDs for all tests, grouped by category.

    Returns
    -------
    tuple
        Category view metadata, category summary tables, category weights, and all
        discovered framework IDs.
    """
    category_views = {}
    category_tables = {}
    category_weights = {}
    framework_ids: set[str] = set()

    # `category` corresponds to the category's directory name
    # We will use the loaded `category_title` for IDs/dictionary keys returned
    for category in all_layouts:
        # Get category name and description
        try:
            with open(APP_ROOT / category / f"{category}.yml") as file:
                category_info = safe_load(file)
                category_title = category_info.get("title", category)
                category_descrip = category_info.get("description", "")
                category_weight = category_info.get("weight", 1)
                benchmark_weights = category_info.get("benchmark_weights", {})
        except FileNotFoundError:
            category_title = category
            category_descrip = ""
            category_weight = 1
            benchmark_weights = {}

        # Build category summary table
        summary_table = build_summary_table(
            all_tables[category],
            table_id=f"{category_title}-summary-table",
            description=category_descrip,
            weights={f"{key} Score": value for key, value in benchmark_weights.items()},
        )

        # Store category weight for overall summary
        category_weights[f"{category_title} Score"] = category_weight

        category_tables[category_title] = summary_table

        # Build weight components for category summary table
        weight_components = build_weight_components(
            header="Benchmark weights",
            table=summary_table,
            column_widths=getattr(summary_table, "column_widths", None),
        )

        test_entries = []
        for test_name in all_layouts[category]:
            framework_id = all_frameworks[category][test_name]
            framework_ids.add(framework_id)
            test_entries.append(
                {
                    "name": test_name,
                    "framework_id": framework_id,
                    "layout": all_layouts[category][test_name],
                }
            )

        category_views[category_title] = {
            "title": category_title,
            "description": category_descrip,
            "summary_table": summary_table,
            "weight_components": weight_components,
            "tests": test_entries,
        }

        # Register benchmark table -> category table callbacks
        # Category summary table columns add "Score" to name for clarity
        for test_name, benchmark_table in all_tables[category].items():
            register_benchmark_to_category_callback(
                benchmark_table_id=benchmark_table.id,
                category_table_id=f"{category_title}-summary-table",
                benchmark_column=test_name + " Score",
                model_name_map=getattr(benchmark_table, "model_name_map", None),
            )

    return category_views, category_tables, category_weights, framework_ids


def build_category_tab_layout(
    category_view: dict[str, object],
) -> Div:
    """
    Build category tab layout.

    Parameters
    ----------
    category_view
        Category metadata including summary table, controls, and benchmark layouts.

    Returns
    -------
    Div
        Category tab layout.
    """
    category_title = category_view["title"]
    category_description = category_view["description"]
    summary_table = category_view["summary_table"]
    weight_components = category_view["weight_components"]
    tests = category_view["tests"]
    benchmark_section = Div([test["layout"] for test in tests])

    return Div(
        [
            H1(category_title),
            H3(category_description),
            summary_table,
            Store(
                id=f"{category_title}-summary-table-computed-store",
                storage_type="session",
                data=summary_table.data,
            ),
            weight_components,
            Div(
                [
                    Div(
                        style={
                            "width": "100%",
                            "height": "1px",
                            "backgroundColor": "#a7adb3",
                        }
                    ),
                ],
                style={"margin": "32px 0 24px"},
            ),
            benchmark_section,
        ]
    )


def build_framework_tab_views(
    category_views: dict[str, dict[str, object]],
    framework_ids: set[str],
) -> dict[str, dict[str, object]]:
    """
    Build framework-focused tab metadata for non-ML-PEG frameworks.

    Parameters
    ----------
    category_views
        Category metadata including benchmark layout components.
    framework_ids
        All framework IDs discovered from benchmark apps.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of framework ID to grouped benchmark layouts by category.
    """
    framework_views: dict[str, dict[str, object]] = {}
    for framework_id in sorted(framework_ids):
        if framework_id == "ml_peg":
            continue

        category_groups = []
        for category_name, category_view in category_views.items():
            tests = [
                test["layout"]
                for test in category_view["tests"]
                if test["framework_id"] == framework_id
            ]
            if tests:
                category_groups.append({"category": category_name, "tests": tests})

        if category_groups:
            config = get_framework_config(framework_id)
            framework_views[framework_id] = {
                "framework_id": framework_id,
                "label": config["label"],
                "logo": config.get("logo"),
                "category_groups": category_groups,
            }
    return framework_views


def build_framework_tab_layout(framework_view: dict[str, object]) -> Div:
    """
    Build a framework-focused tab containing duplicate benchmark sections.

    Parameters
    ----------
    framework_view
        Framework tab metadata with grouped benchmark layouts by category.

    Returns
    -------
    Div
        Framework tab layout.
    """
    framework_label = framework_view["label"]
    category_groups = framework_view["category_groups"]

    sections = []
    for group in category_groups:
        sections.append(H3(group["category"], style={"marginTop": "26px"}))
        sections.append(Div(group["tests"]))

    return Div(
        [
            H1(f"{framework_label} Benchmarks"),
            Div(
                (
                    "These benchmark sections are duplicates of the category tabs for "
                    "easier collection. Benchmark controls and weights stay in sync."
                ),
                style={
                    "fontSize": "13px",
                    "fontStyle": "italic",
                    "color": "#64748b",
                    "marginTop": "8px",
                    "marginBottom": "8px",
                },
            ),
            *sections,
        ]
    )


def build_summary_table(
    tables: dict[str, DataTable],
    table_id: str = "summary-table",
    description: str | None = None,
    weights: dict[str, float] | None = None,
) -> DataTable:
    """
    Build summary table from a set of tables.

    Parameters
    ----------
    tables
        Dictionary of tables to be summarised.
    table_id
        ID of table being built. Default is 'summary-table'.
    description
        Description of summary table. Default is None.
    weights
        Weights for each column. Default is `None`, which sets all weights to 1.

    Returns
    -------
    DataTable
        Summary table with scores from tables being summarised.
    """
    summary_data = {}
    category_columns = []  # Track all category columns
    for category_name, table in tables.items():
        # Prepare rows for all current models
        if not summary_data:
            summary_data = {model: {} for model in MODELS}

        category_col = f"{category_name} Score"
        category_columns.append(category_col)

        table_name_map = getattr(table, "model_name_map", {}) or {}
        for row in table.data:
            # Category tables may include models not to be included
            # Table headings are of the form "[category] Score"
            # ``original_name`` refers to the original model identifier
            # (no display suffix)
            original_name = table_name_map.get(row["MLIP"], row["MLIP"])
            if original_name in summary_data:
                summary_data[original_name][category_col] = row["Score"]

    # Ensure all models have entries for all category columns (None if missing)
    data = []
    for mlip in summary_data:
        row = {"MLIP": mlip}
        for category_col in category_columns:
            row[category_col] = summary_data[mlip].get(category_col, None)
        data.append(row)

    data = calc_table_scores(data, weights=weights)

    columns_headers = ("MLIP",) + tuple(key + " Score" for key in tables) + ("Score",)

    columns = [{"name": headers, "id": headers} for headers in columns_headers]
    tooltip_header = {
        header + " Score": table.description for header, table in tables.items()
    }

    for column in columns:
        column_id = column["id"]
        if column_id != "MLIP":
            column["type"] = "numeric"
            column["format"] = sig_fig_format()

    style = get_table_style(data)
    registry_configs = load_model_registry_configs()
    row_models: list[str] = []
    for row in data:
        mlip = row.get("MLIP")
        if isinstance(mlip, str) and mlip not in row_models:
            row_models.append(mlip)
    model_configs = {mlip: (registry_configs.get(mlip) or {}) for mlip in row_models}
    model_levels = {
        mlip: (model_configs[mlip].get("level_of_theory")) for mlip in row_models
    }
    warning_styles, tooltip_rows = build_level_of_theory_warnings(
        data,
        model_levels,
        {},
        model_configs,
    )
    style_with_warnings = style + warning_styles

    # Calculate column widths based on column names
    calculated_widths = calculate_column_widths(columns_headers)
    # Limit max width to 150px for better wrapping on long column names
    column_widths = {
        col_id: min(width, 150) for col_id, width in calculated_widths.items()
    }

    style_cell_conditional = []
    for column_id, width in column_widths.items():
        col_width = f"{width}px"
        alignment = "left" if column_id == "MLIP" else "center"
        style_cell_conditional.append(
            {
                "if": {"column_id": column_id},
                "width": col_width,
                "minWidth": col_width,
                "maxWidth": col_width,
                "textAlign": alignment,
            }
        )

    tooltip_header["Score"] = "Weighted average of scores (higher is better)"

    table = DataTable(
        data=data,
        columns=columns,
        id=table_id,
        sort_action="native",
        style_data_conditional=style_with_warnings,
        style_cell_conditional=style_cell_conditional,
        style_header={
            "whiteSpace": "normal",
            "height": "auto",
            "minHeight": "70px",
            "textAlign": "center",
            "verticalAlign": "middle",
            "lineHeight": "1.4",
            "padding": "8px",
        },
        style_header_conditional=[
            {
                "if": {"column_id": "MLIP"},
                "textAlign": "left",
            }
        ],
        tooltip_data=tooltip_rows,
        tooltip_delay=100,
        tooltip_duration=None,
        persistence=True,
        persistence_type="session",
        persisted_props=["data"],
        tooltip_header=tooltip_header,
        editable=False,
    )
    table.column_widths = column_widths
    table.description = description
    table.model_levels_of_theory = model_levels
    table.metric_levels_of_theory = {}
    table.model_configs = model_configs
    table.weights = weights
    return table


def build_tabs(
    full_app: Dash,
    category_views: dict[str, dict[str, object]],
    framework_tab_views: dict[str, dict[str, object]],
    summary_table: DataTable,
    weight_components: Div,
) -> None:
    """
    Build tab layouts and summary tab.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    category_views
        Category metadata required to render tab content.
    framework_tab_views
        Framework tab metadata for additional non-ML-PEG frameworks.
    summary_table
        Summary table with score from each category.
    weight_components
        Weight sliders, text boxes and reset button.
    """
    framework_tabs = []
    for framework_id in sorted(framework_tab_views):
        framework_view = framework_tab_views[framework_id]
        tab_label: str | Div = framework_view["label"]
        logo = framework_view.get("logo")
        if isinstance(logo, str) and logo:
            tab_label = Div(
                [
                    Img(
                        src=logo,
                        alt=f"{framework_view['label']} logo",
                        style={
                            "width": "14px",
                            "height": "14px",
                            "borderRadius": "50%",
                            "objectFit": "cover",
                        },
                    ),
                    Span(framework_view["label"]),
                ],
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "gap": "6px",
                },
            )
        framework_tabs.append(
            Tab(
                label=tab_label,
                value=f"framework-{framework_id}",
            )
        )
    all_tabs = (
        [Tab(label="Summary", value="summary-tab", id="summary-tab")]
        + [
            Tab(label=category_name, value=category_name)
            for category_name in category_views
        ]
        + framework_tabs
    )

    tabs_layout = [
        build_onboarding_modal(),
        build_tutorial_button(),
        Div(
            [
                H1("ML-PEG"),
                Tabs(id="all-tabs", value="summary-tab", children=all_tabs),
                Loading(
                    Div(id="tabs-content"),
                    type="circle",
                    color="#119DFF",
                    fullscreen=False,
                    # dont trigger both start up load wheel + tab change load wheel
                    # (when switching to summary tab)
                    target_components={"tabs-content": "children"},
                    style={
                        # Pin near the top so the spinner is visible on long pages
                        # (default is centre of page)
                        "position": "fixed",
                        "top": "300px",
                        "left": "50%",
                        "transform": "translateX(-50%)",
                        "zIndex": "1100",
                    },
                    parent_style={"position": "relative"},
                ),
            ],
            style={"flex": "1", "marginBottom": "40px"},
        ),
        build_footer(),
    ]

    full_app.layout = Div(
        tabs_layout,
        style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"},
    )

    @callback(
        Output("tabs-content", "children"),
        Input("all-tabs", "value"),
    )
    def select_tab(tab: str) -> Div:
        """
        Select tab contents to be displayed.

        Parameters
        ----------
        tab
            Name of tab selected.

        Returns
        -------
        Div
            Summary or tab contents to be displayed.
        """
        if tab == "summary-tab":
            return Div(
                [
                    H1("Benchmarks Summary"),
                    summary_table,
                    weight_components,
                    Store(
                        id="summary-table-scores-store",
                        storage_type="session",
                    ),
                    build_faqs(),
                ]
            )
        if tab.startswith("framework-"):
            framework_id = tab.removeprefix("framework-")
            return Div([build_framework_tab_layout(framework_tab_views[framework_id])])
        return Div([build_category_tab_layout(category_views[tab])])


def build_full_app(full_app: Dash, category: str = "*") -> None:
    """
    Build full app layout and register callbacks.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    """
    # Get layouts and tables for each test, grouped by categories
    all_layouts, all_tables, all_frameworks = get_all_tests(category=category)

    if not all_layouts:
        raise ValueError("No tests were built successfully")

    # Combine tests into categories and create category summary
    cat_views, cat_tables, cat_weights, framework_ids = build_category(
        all_layouts, all_tables, all_frameworks
    )
    framework_tab_views = build_framework_tab_views(cat_views, framework_ids)
    # Build overall summary table
    summary_table = build_summary_table(cat_tables, weights=cat_weights)
    weight_components = build_weight_components(
        header="Category weights",
        table=summary_table,
        column_widths=summary_table.column_widths,
    )
    # Build summary and category tabs
    build_tabs(
        full_app,
        cat_views,
        framework_tab_views,
        summary_table,
        weight_components,
    )
    register_onboarding_callbacks()
