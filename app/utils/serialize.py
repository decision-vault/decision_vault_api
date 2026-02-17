from bson import ObjectId


def oid_to_str(value):
    if isinstance(value, ObjectId):
        return str(value)
    return value


def serialize_doc(doc: dict) -> dict:
    return {k: oid_to_str(v) for k, v in doc.items()}
