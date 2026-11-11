
from app.core.config import SearchConfig


def normalize_value(value):
    """Convert a field value into clean text."""

    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        )

    return " ".join(str(value).split())


def prepare_document(record: dict, config: SearchConfig) -> dict:
    """Convert an application record into a searchable document."""

    record_id = record.get(config.id_field)

    if record_id is None:
        raise ValueError(
            f"Missing ID field: {config.id_field}"
        )

    fields = {}

    for field_name in config.searchable_fields:

        normalized_value = normalize_value(record.get(field_name))

        if normalized_value:
            fields[field_name] = normalized_value

    return {
        "id": str(record_id),
        "text": " ".join(fields.values()),
        "fields": fields
    }
