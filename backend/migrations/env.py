"""Alembic setup. Every module's models.py must be imported here, so autogenerate sees it."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

import app.core.models  # noqa: F401
import app.modules.knowledge.models  # noqa: F401
from app.config import get_settings
from app.core.database import Base

# Use the logging setup from alembic.ini.
if context.config.config_file_name:
    fileConfig(context.config.config_file_name)

# One PostgreSQL schema per module (vault: Data model)
SCHEMAS = {"core", "knowledge"}


# Only look at our own schemas, so autogenerate ignores public and the extensions.
def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    if type_ == "schema":
        return name in SCHEMAS
    return True


# Connect with the same database URL as the app and run the pending migrations.
def run_migrations_online() -> None:
    engine = create_engine(get_settings().db_url)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            include_schemas=True,
            include_name=include_name,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
