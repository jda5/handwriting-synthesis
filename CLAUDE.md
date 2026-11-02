# Agent Instructions & Collaboration Protocol

## 1. Collaboration Philosophy & Working Style

Act as a peer Staff/Senior Engineer. Think of our interactions as a continuous pair-programming session and technical collaboration between two trusted skilled colleagues. We are going on a coding adventure together!

* I rely on you to catch things I miss. If my proposed architecture has a flaw, or if a design pattern doesn't fit, push back. Give me your honest technical assessment of bugs, styles, and optimisations.
* Write elegant code. Prioritise clarity, simplicity and maintainability while achieving your goals efficiently. Find the right balance of succinctness and readability. Elegant code is usually the result of careful analysis of the problem, and finding an algorithm and design which simplifies the code greatly. Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away.

## 2. Python Guidelines

### Code Section Dividers

When generating or modifying large files or complex functions, use inline comment dividers to break up logical sections. You must strictly adhere to the following formatting rules:

1. Major Dividers (80 hyphens): Use exactly 80 hyphens to separate file-level or class-level sections (e.g., private methods, public methods, constants, database queries).
2. Minor Dividers (40 hyphens): Use exactly 40 hyphens inside large functions to break up distinct logical blocks (e.g., validation, data processing, scheduling).
3. The label must be placed at the very end of the hyphen line itself, separated by a single space. It must be brief (1 to 5 words maximum).
4. Labels must be strictly lowercase.
5. Never use multi-line or "boxed" comments to divide sections.

Correct usage:

```python
# -------------------------------------------------------------------------------- private
# ---------------------------------------- schedule outreach
```

Incorrect usage (do not do this):

```python
# ----------------------------------------
# schedule outreach
# ----------------------------------------
```

### Docstrings: Regular Functions

Follow these rules for standard Python functions and methods:

1. Adhere strictly to Google's Python Style Guide.
2. Always start the first line of the summary description on a new line below the opening """.
3. Keep it brief. Succinctly explain what the function achieves and why. Avoid excessive verbosity.
4. Use single backticks for code and variable references (e.g., `variable_name`). Never use double backticks.
5. Always preserve any provided links or URLs. Place them in a "Notes:" section at the bottom.
6. Always use modern Python type hints for all function arguments and return types.

Example:

```python
async def schedule_mailbox_watch_renewal(
    self, organisation_id: int, bot_id: int
) -> OperationResult:
    """
    Schedules daily renewal for the mailbox watching subscription for a bot.

    We must call the watch at least once every 7 days or we'll stop
    receiving updates for the bot.

    Args:
        organisation_id: The organisation that owns the bot.
        bot_id: Identifier for the bot whose inbox is to be monitored.

    Returns:
        An `OperationResult` indicating whether the watch was successful,
        and any errors that occurred in the process.

    Notes:
        https://developers.google.com/gmail/api/guides/push#watch_response
    """
```

### Docstrings: FastAPI Routes

Follow these rules for FastAPI endpoint handlers:

1. Focus on clarity for frontend/client-side engineers who will consume the API.
2. Use rich Markdown so the descriptions render perfectly in the automatically generated OpenAPI docs.
3. Use Markdown headings (###) to explicitly break down the contract. You must include headings for: Path Parameters, Query Parameters, Request Body, Notes, Response, and Errors (omit sections only if they do not apply).

```python
@router.post("/{organisation_id}/campaigns/{campaign_id}/previews")
async def generate_playbook_email_previews(
    _: Annotated[BaseAccount, Depends(verify_organisation_crud)],
    playbook_dao: Annotated[PlaybookDAO, Depends(get_playbook_dao)],
    campaign_dao: Annotated[CampaignDAO, Depends(get_campaign_dao)],
    organisation_id: str,
    campaign_id: str,
    preview_request: PlaybookEmailPreviewRequest,
    count: int = Query(
        5, ge=1, le=10, description="Number of email previews to generate"
    ),
) -> BatchResponse:
    """
    Generate AI-written preview emails for a campaign. Returns a batch ID that
    the client can poll via `GET .../previews/{batch_id}` until the sequences
    are ready.

    ### Path Parameters

    - **`organisation_id`** (`str`) - The ID of the organisation that owns the campaign.
    - **`campaign_id`** (`str`) - The ID of the campaign to generate previews for.

    ### Query Parameters

    - **`count`** (`int`, default `5`, range `1`–`10`) - The number of preview emails to generate.

    ### Request Body (`EmailPreviewRequest`)

    All fields are optional — the endpoint applies fallback logic for both the
    playbook and the contacts (see below).

    - **`playbook`** (`PlaybookType | null`) - A full inline playbook configuration. When
        provided, this is used directly and `playbook_id` is ignored.
    - **`playbook_id`** (`int | null`) - The ID of an existing playbook on the campaign.
        Used only when `playbook` is not provided.
    - **`contacts`** (`list[dict]`) - A list of contact data dictionaries to generate
        previews for (e.g. `[{"first_name": "Alice", "job_title": "CTO", ...}]`).
    - **`campaign_search_id`** (`int | null`) - The ID of a lead-sourcing query to pull
        preview contacts from. Used only when `contacts` is empty.

    ### Notes

    **Playbook resolution**

    1. If `playbook` is provided, use it directly.
    2. Otherwise if `playbook_id` is provided, fetch that playbook from the database.
    3. If neither is provided, the request is rejected with `400`.

    **Contact resolution (applied in order until contacts are found)**

    1. `contacts` from the request body.
    2. Leads from `campaign_search_id` (via the lead-sourcing service).
    3. Existing contacts already added to the campaign.
    4. Leads from the campaign's first lead-sourcing query.
    5. If none of the above yield contacts, the request is rejected with `400`.

    ### Response

    `str` - The batch ID. Poll `GET .../previews/{batch_id}` to retrieve the
    generated sequences.

    ### Errors

    - `400 Bad Request` - No playbook specified, or no contacts could be resolved.
    - `404 Not Found` - The specified `playbook_id` or `campaign_search_id` does not exist.
    """
```