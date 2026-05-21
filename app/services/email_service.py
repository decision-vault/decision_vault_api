from __future__ import annotations

import smtplib
from email.message import EmailMessage
import logging

from app.core.config import settings

logger = logging.getLogger("decisionvault.email")


def send_email(*, to_email: str, subject: str, text_body: str) -> None:
    if not settings.smtp_username or not settings.smtp_password:
        raise RuntimeError("SMTP credentials not configured")

    smtp_password = (settings.smtp_password or "").strip().replace(" ", "")
    smtp_username = (settings.smtp_username or "").strip()
    smtp_host = (settings.smtp_host or "").strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_from_email}>"
    msg["To"] = to_email
    msg.set_content(text_body)

    try:
        with smtplib.SMTP(smtp_host, settings.smtp_port, timeout=20) as server:
            if settings.smtp_use_starttls:
                server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
    except Exception:
        logger.exception("smtp_send_failed to=%s subject=%s", to_email, subject)
        raise


def send_org_invite_email(
    *,
    to_email: str,
    invite_link: str,
    inviter_email: str | None,
    role: str,
    org_name: str | None,
    projects: list[str] | None = None,
) -> None:
    org_label = org_name or "your DecisionVault organization"
    inviter_label = inviter_email or "an admin"
    subject = f"You've been invited to {org_label}"
    projects_line = ""
    if projects:
        safe = [p for p in projects if p]
        if safe:
            projects_line = "\n\nProject access:\n" + "\n".join(f"- {name}" for name in safe)
    text_body = (
        f"{inviter_label} invited you to join {org_label} as {role}.\n\n"
        f"{projects_line}"
        f"Accept invite: {invite_link}\n\n"
        "If you weren't expecting this, you can ignore this email."
    )
    send_email(to_email=to_email, subject=subject, text_body=text_body)
