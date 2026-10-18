"""ResolveAI UI. Run with: streamlit run app.py.

Navigation and chat rendering are separate so future ticket views can be added
without changing the retrieval workflow. History is display-only session state;
it is not passed to the bot as conversation memory.
"""
from __future__ import annotations

import os
import time
from uuid import uuid4

import streamlit as st

from ticket_store import (
    TicketStoreError, create_ticket, get_ticket, init_db, list_tickets,
    update_ticket_status,
)

TICKET_STATUSES = ("Open", "In Progress", "Resolved")

APP_TITLE = "ResolveAI – Intelligent Customer Support Platform"

# Suggestions are grounded in existing FAQ, manual, and historical ticket topics.
HELP_CATEGORIES = [
    ("🔑", "Account & Sign In", "Get back into your account.", [
        "How do I reset my password?",
        "My 2FA codes stopped working after I got a new phone",
    ]),
    ("🛡️", "Security & Privacy", "Protect your account and data.", [
        "How do I enable two-factor authentication?",
        "Where is my data stored and is it encrypted?",
    ]),
    ("📁", "File Sharing", "Share files and fix access issues.", [
        "Can I share files with people who don't have CloudSync Pro?",
        "Why does my shared link show a 404 for an external recipient?",
    ]),
    ("💳", "Billing & Plans", "Understand plans and billing.", [
        "What are the storage limits for each plan?",
        "Why was I charged twice in the same month?",
    ]),
    ("🔧", "Troubleshooting", "Find help for uploads and sync.", [
        "Sync is stuck at Preparing to upload on Windows. What should I do?",
        "What's the max file size I can upload from the website?",
    ]),
    ("💬", "Talk to Support", "Understand when human help is needed.", []),
]


def apply_styles() -> None:
    # Only trusted, static markup is rendered as HTML.
    st.markdown("""
<style>
.stApp { background: #f6f8fb; color: #142d49; color-scheme: light;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp p, .stApp button, .stApp input, .stApp textarea {
    font-family: inherit; font-size: 15px; line-height: 1.5;
}
.stApp h1 { font-size: 29px; font-weight: 600; line-height: 1.3; }
.stApp h4 { font-size: 18px; font-weight: 600; line-height: 1.35; }
.stApp h2, .stApp h3 { font-size: 22px; font-weight: 600; line-height: 1.3; }
[data-testid="stHeader"] { background: #f6f8fb; }
[data-testid="stMainBlockContainer"], .main .block-container {
    max-width: 1080px; padding-top: 4rem; padding-bottom: 1.5rem;
}
h1, h2, h3, [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] { color: #142d49; }
[data-testid="stCaptionContainer"] { color: #4a6278; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #dde5ee; }
/* Match the main content inset while leaving Streamlit’s toolbar room. */
[data-testid="stSidebarUserContent"] { padding-top: 4rem; }
.resolve-brand { display: flex; align-items: center; gap: .6rem;
    color: #142d49; font-size: 26px; font-weight: 700; }
.resolve-icon { display: inline-grid; place-items: center; background: #e0f3f1;
    color: #096b6d; border: 1px solid #b7deda; border-radius: 14px;
    width: 48px; height: 48px; font-size: 1.8rem; }
.resolve-welcome { padding: 0 0 .25rem; }
.resolve-welcome h1 { color: #142d49; font-size: 29px; font-weight: 600;
    margin: 0; padding: 0 0 .5rem; line-height: 1.3; }
.resolve-welcome p { color: #4a6278; margin-bottom: 0; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #e6f4f2; color: #12515a; border-color: #087f80;
    border-left: 3px solid #087f80; font-weight: 600;
}
.resolve-category { min-height: 88px; }
.resolve-category h4 { font-size: 19px; font-weight: 600; line-height: 1.35;
    margin: 0 0 .5rem; padding: 0; }
.resolve-category p { color: #4a6278; font-size: 14px; margin: 0; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 14px; }
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #dce5ef;
    border-radius: 16px; margin-bottom: 1rem; padding: 1.1rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #eaf5f4; border-color: #bededa;
}
[data-testid="stExpander"] { background: #fff; border-radius: 12px; color: #142d49; }
[data-testid="stExpander"] summary { color: #142d49; }
.stButton > button { background: #fff; color: #153c52; border: 1px solid #9ebcc7;
    border-radius: 9px; min-height: 2.6rem; white-space: normal; }
.stButton > button p { color: inherit !important; }
.stButton > button:hover:enabled { background: #e6f4f2; color: #07595d; border-color: #087f80; }
.stButton > button:focus-visible { outline: 3px solid #087f80; outline-offset: 2px; }
.stButton > button:disabled { background: #f0f3f6; color: #586777;
    border-color: #d2dae2; opacity: 1; cursor: default; }
[data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea {
    background: #fff; color: #142d49; -webkit-text-fill-color: #142d49;
}
[data-testid="stChatInput"] { background: #fff; border-color: #9ebcc7; border-radius: 14px; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"], .main .block-container { padding: 4rem 1rem 1rem; }
    [data-testid="stHorizontalBlock"] { flex-direction: column; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important; flex: 1 1 100% !important;
    }
    [data-testid="stChatMessage"] { padding: .75rem; }
}
</style>
""", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="resolve-brand"><span class="resolve-icon">◇</span> ResolveAI</div>',
                    unsafe_allow_html=True)
        st.caption("Intelligent Customer Support Platform")
        st.divider()
        for page, icon in (
            ("AI Support Chat", "\U0001f4ac"),
            ("My Tickets", "\U0001f3ab"),
            ("Support Dashboard", "\U0001f4ca"),
        ):
            if st.button(
                page, icon=icon, key=f"nav_{page}", use_container_width=True,
                type="primary" if st.session_state.selected_page == page else "secondary",
            ):
                st.session_state.selected_page = page
                st.rerun()



def render_welcome() -> None:
    welcome, action = st.columns([4, 1])
    with action:
        if st.button("+ New Chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.pop("pending_query", None)
            st.session_state.help_topics_expanded = False
            st.rerun()
    with welcome:
        st.markdown("""
<div class="resolve-welcome">
    <h1>How can we help you today?</h1>
    <p>Ask ResolveAI about account access, billing, security, file sharing, and troubleshooting.</p>
</div>
""", unsafe_allow_html=True)


def render_quick_help(connected: bool) -> None:
    st.subheader("Quick Help")
    st.caption("Explore a category, then choose a question to ask ResolveAI.")
    for row in range(0, len(HELP_CATEGORIES), 3):
        for col, (icon, title, description, questions) in zip(
            st.columns(3), HELP_CATEGORIES[row:row + 3]
        ):
            with col, st.container(border=True):
                st.markdown(
                    f'<div class="resolve-category"><h4>{icon} {title}</h4>'
                    f'<p>{description}</p></div>', unsafe_allow_html=True,
                )
                with st.expander("Suggested questions" if questions else "About human support"):
                    if questions:
                        for question in questions:
                            if st.button(question, key=f"help_{question}",
                                         disabled=not connected, use_container_width=True):
                                st.session_state.pending_query = question
                                st.session_state.help_topics_expanded = False
                                st.rerun()
                    else:
                        st.write("Describe your issue in the chat. ResolveAI checks the knowledge base and returns an escalation response when it cannot answer confidently.")
                        st.caption("After an escalation, you can create a local support ticket. No agent is notified automatically; use My Tickets to look up its current status.")


def render_sources(trace: dict) -> None:
    """Keep source documents separate from routing details; avoid nested expanders."""
    scored = trace.get("scored") or []
    with st.expander(f"Supporting sources · {len(scored)} retrieved documents"):
        st.caption("Retrieved candidates may not support an answer, especially when the question is escalated.")
        for doc, sim in scored:
            st.markdown(f"**[{doc.metadata['id']}] {doc.metadata['title']}**")
            st.caption(f"{doc.metadata['source']} · Similarity {sim:.2f}")
            st.write(doc.page_content)
            st.divider()
        if not scored:
            st.caption("No source documents were retrieved.")
    with st.expander("Confidence & reasoning trace"):
        st.write(f"**Decision:** {trace.get('decision', 'unknown')}")
        st.write(f"**Top retrieval similarity:** {trace.get('top_sim', 0.0):.2f}")
        st.caption("Similarity measures the context match, not the probability that an answer is correct.")
        st.write(f"**LLM review:** {'Used' if trace.get('judge_used') else 'Not used'}")
        if trace.get("escalation_reason"):
            st.write(f"**Escalation reason:** {trace['escalation_reason']}")
        st.caption("Routing: ≥ 0.55 → answer; 0.35–0.55 → LLM review; < 0.35 → escalate. The answer model can also flag insufficient context.")


def render_ticket_option(turn: dict) -> None:
    """Keep each form and confirmation attached to its own escalation turn."""
    if turn.get("ticket_id"):
        st.success("Support ticket created - Initial status: Open")
        st.write(f"**Ticket ID:** `{turn['ticket_id']}`")
        st.caption("Saved locally. Keep this ID for reference. No agent has been notified; use My Tickets to look up its current status.")
        return

    # Stored on the history entry, so keys survive reruns and never collide
    # with another escalation, including after New Chat.
    turn.setdefault("ticket_form_id", str(uuid4()))
    key = turn["ticket_form_id"]
    if not turn.get("ticket_form_open"):
        if st.button("Create Support Ticket", key=f"create_{key}"):
            turn["ticket_form_open"] = True
            st.rerun()
        return

    with st.form(f"ticket_form_{key}"):
        st.markdown("**Create Support Ticket**")
        customer_name = st.text_input("Customer name", key=f"name_{key}")
        # Read-only review preserves the original question exactly, even for
        # older escalations displayed alongside a newer customer question.
        issue_description = st.text_area(
            "Issue description", value=turn["original_query"],
            disabled=True, key=f"issue_{key}",
        )
        st.caption("This saves a local ticket only; it does not notify a human agent.")
        submitted = st.form_submit_button("Submit Support Ticket")
    if not submitted:
        return
    if not customer_name.strip() or not issue_description.strip():
        st.error("Please provide a customer name and a nonempty issue description.")
        return
    # A saved ID above suppresses all future inserts for this escalation.
    try:
        init_db()
        ticket_id = create_ticket(
            customer_name.strip(), turn["original_query"],
            turn["trace"].get("escalation_reason", "Human review recommended."),
        )
    except TicketStoreError:
        st.error("Your ticket could not be saved. Please try again later. No ticket creation has been confirmed.")
        return
    turn["ticket_id"] = ticket_id
    turn["ticket_form_open"] = False
    st.rerun()


def render_answer(
    content: str, trace: dict, elapsed: float | None = None,
    turn: dict | None = None,
) -> None:
    if trace.get("decision") == "escalate":
        st.warning("Human support recommended - more context or review is needed.")
        # Render accurate UI copy instead of the backend's automatic handoff
        # promise, without rewriting customer text or source documents.
        st.write("I could not find enough supporting information to answer confidently. You may create a local support ticket below and keep its ID for reference. No human agent is contacted automatically; use My Tickets to look up its current status.")
        if turn is not None:
            st.write("**Your question:**")
            st.write(turn["original_query"])
            render_ticket_option(turn)
    else:
        st.write(content)
    if elapsed is not None:
        st.caption(f"Responded in {elapsed:.1f}s")
    render_sources(trace)


@st.cache_resource(show_spinner="Preparing the support knowledge base…")
def get_bot():
    from bot import build_bot
    return build_bot()


def render_chat(bot) -> None:
    st.subheader("Your conversation")
    st.caption("Answers include source references. Use + New Chat above to start a new conversation.")
    # Auto-height containers keep history and composer in the normal page flow.
    messages = st.container()
    with messages:
        original_query = ""
        for turn in st.session_state.history:
            if turn["role"] == "user":
                original_query = turn["content"]
            elif turn.get("trace"):
                # Backfill history created before ticket creation was added.
                turn.setdefault("original_query", turn["trace"].get("query", original_query))
            with st.chat_message(turn["role"]):
                st.caption("You" if turn["role"] == "user" else "ResolveAI · Support assistant")
                if turn.get("trace"):
                    render_answer(turn["content"], turn["trace"], turn.get("elapsed"), turn)
                else:
                    st.write(turn["content"])

    query = st.chat_input("Ask ResolveAI a question...")
    # Consume queued suggestions once, even if a typed question also arrived.
    pending_query = st.session_state.pop("pending_query", None)
    query = query or pending_query
    if not query:
        return
    st.session_state.help_topics_expanded = False
    st.session_state.history.append({"role": "user", "content": query})
    with messages, st.chat_message("user"):
        st.caption("You")
        st.write(query)
    with messages, st.chat_message("assistant"):
        st.caption("ResolveAI · Support assistant")
        with st.spinner("Finding relevant sources and preparing your response…"):
            t0 = time.time()
            try:
                result = bot.ask(query)
            except Exception:
                # Provider exception strings may contain sensitive request details.
                st.error("We couldn’t complete your request. Check your connection and API configuration, then try again.")
                st.stop()
            elapsed = time.time() - t0
        answer = result.get("answer", "")
        turn = {
            "role": "assistant", "content": answer, "trace": result,
            "elapsed": elapsed, "original_query": query,
        }
        st.session_state.history.append(turn)
        render_answer(answer, result, elapsed, turn)

    # Refresh the initial grid/topic browser after the completed exchange.
    # The queued query has already been popped, so this cannot resubmit it.
    st.rerun()


def render_ticket_details(ticket: dict, *, administrative: bool = False) -> None:
    """Read-only details shared by customer lookup and the admin dashboard."""
    with st.container(border=True):
        st.markdown("#### Customer information")
        customer, status = st.columns([2, 1])
        with customer:
            st.write("**Customer name:**", ticket["customer_name"])
            st.write("**Ticket ID:**", ticket["ticket_id"])
            st.write("**Created (UTC):**", ticket["created_at"])
        with status:
            st.markdown("**Current status**")
            st.write(ticket["status"])
        st.divider()
        st.markdown("#### Issue description")
        st.text(ticket["issue_description"])
        if administrative:
            st.divider()
            st.markdown("#### Escalation reason")
            st.text(ticket["escalation_reason"])


def clear_ticket_search() -> None:
    """Reset lookup state before widgets render; never touch stored tickets."""
    st.session_state.ticket_lookup_input = ""
    st.session_state.pop("ticket_lookup_id", None)


def submit_ticket_search() -> None:
    st.session_state.ticket_lookup_id = st.session_state.ticket_lookup_input.strip()


def render_my_tickets() -> None:
    st.title("My Tickets")
    st.info("Local demonstration only: a ticket ID lookup is not secure customer authentication. Anyone with a ticket ID can view its details.")
    # Restore a submitted search if Streamlit cleaned up the hidden widget
    # while another page was selected. The durable lookup ID is not a widget.
    st.session_state.setdefault(
        "ticket_lookup_input", st.session_state.get("ticket_lookup_id", ""),
    )
    with st.form("ticket_lookup"):
        st.text_input("Ticket ID", key="ticket_lookup_input")
        find, clear = st.columns(2)
        with find:
            search = st.form_submit_button(
                "Find Ticket", on_click=submit_ticket_search, use_container_width=True,
            )
        with clear:
            st.form_submit_button(
                "Clear Search", on_click=clear_ticket_search, use_container_width=True,
            )
    if search and not st.session_state.get("ticket_lookup_id"):
        st.warning("Please enter a ticket ID.")
    lookup_id = st.session_state.get("ticket_lookup_id")
    if not lookup_id:
        return
    try:
        init_db()
        ticket = get_ticket(lookup_id)
    except TicketStoreError:
        st.error("We could not look up your ticket. Please try again later.")
        return
    if ticket is None:
        st.info("No ticket was found with that ID. Check the ID and try again.")
    else:
        render_ticket_details(ticket)


def render_support_dashboard() -> None:
    st.title("Support Dashboard")
    st.warning("Administrative demonstration without authentication. This dashboard exposes all local tickets and permits status changes. It is not secure for public use.")
    try:
        init_db()
        tickets = list_tickets()
    except TicketStoreError:
        st.error("Ticket storage is unavailable. Please try again later.")
        return
    if notice := st.session_state.pop("ticket_update_notice", None):
        st.success(notice)
    st.subheader("Ticket Overview")
    counts = [("Total Tickets", len(tickets))] + [
        (status, sum(ticket["status"] == status for ticket in tickets))
        for status in TICKET_STATUSES
    ]
    for column, (label, count) in zip(st.columns(4), counts):
        with column, st.container(border=True):
            st.metric(label, count)
    st.divider()
    st.subheader("Manage Support Tickets")
    status_filter = st.selectbox(
        "Filter by status", ("All statuses", *TICKET_STATUSES), key="ticket_status_filter",
    )
    filtered = [ticket for ticket in tickets
                if status_filter == "All statuses" or ticket["status"] == status_filter]
    if not filtered:
        st.info("No tickets match this filter." if tickets else "No support tickets have been created yet.")
        return
    # Abbreviations are presentation-only; selection and updates use full IDs.
    rows = [{
        "Ticket": ticket["ticket_id"][:8] + "...",
        "Customer": ticket["customer_name"],
        "Issue summary": " ".join(ticket["issue_description"].split())[:90],
        "Status": ticket["status"],
        "Created (UTC)": ticket["created_at"],
    } for ticket in filtered]
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "Ticket": st.column_config.TextColumn("Ticket", width="small"),
            "Customer": st.column_config.TextColumn("Customer", width="medium"),
            "Issue summary": st.column_config.TextColumn("Issue summary (preview)", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Created (UTC)": st.column_config.TextColumn("Created (UTC)", width="medium"),
        },
    )
    st.caption("Ticket IDs and issue descriptions are abbreviated above. Select a ticket below to see its full details.")
    st.divider()
    st.subheader("Ticket Details & Status Management")
    by_id = {ticket["ticket_id"]: ticket for ticket in filtered}
    # A filter-specific key prevents a selection from pointing outside the list.
    selection_key = f"dashboard_ticket_{status_filter}"
    if st.session_state.get(selection_key) not in by_id:
        st.session_state[selection_key] = next(iter(by_id))
    selected_id = st.selectbox(
        "Select a ticket", list(by_id), key=selection_key,
    )
    ticket = by_id[selected_id]
    render_ticket_details(ticket, administrative=True)
    with st.form(f"status_update_{selected_id}_{ticket['status']}"):
        st.markdown("#### Update ticket status")
        st.caption("Choose a status, then select Update Status to save the change.")
        new_status = st.selectbox(
            "New status", TICKET_STATUSES, index=TICKET_STATUSES.index(ticket["status"]),
        )
        submitted = st.form_submit_button("Update Status")
    if submitted:
        try:
            updated = update_ticket_status(selected_id, new_status)
        except (TicketStoreError, ValueError):
            st.error("The status update could not be confirmed. Please refresh the dashboard and try again.")
            return
        if not updated:
            st.error("This ticket could not be found. Refresh the dashboard to reload the ticket list.")
            return
        st.session_state.ticket_update_notice = f"Ticket {selected_id} updated to {new_status}."
        # Reload all tickets and global counts, including when the updated
        # ticket no longer belongs to the selected status filter.
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💠", layout="wide")
    apply_styles()
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.setdefault("help_topics_expanded", False)
    st.session_state.setdefault("selected_page", "AI Support Chat")
    render_sidebar()
    if st.session_state.selected_page == "My Tickets":
        render_my_tickets()
        return
    if st.session_state.selected_page == "Support Dashboard":
        render_support_dashboard()
        return
    from knowledge_base import load_documents
    load_documents()
    render_welcome()
    connected = bool(os.environ.get("OPENAI_API_KEY"))
    if not st.session_state.history:
        render_quick_help(connected)
    else:
        label = ("Collapse Help Topics" if st.session_state.help_topics_expanded
                 else "Browse Help Topics")
        if st.button(label, key="browse_help_topics"):
            st.session_state.help_topics_expanded = not st.session_state.help_topics_expanded
            st.rerun()
        if st.session_state.help_topics_expanded:
            render_quick_help(connected)
    if not connected:
        st.info("Configure OPENAI_API_KEY in the application environment to start asking questions.")
        st.stop()
    # A nested chat input is inline, rather than pinned to the viewport bottom.
    with st.container(border=True):
        render_chat(get_bot())


if __name__ == "__main__":
    main()
