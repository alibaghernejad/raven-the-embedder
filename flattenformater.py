from typing import Dict, Any, List

def flatten_json(doc: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in doc.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {value}")
    return "\n".join(parts)

def flatten_json_gapfilm_full(doc: Dict[str, Any]) -> str:
    root = doc[0]
    src  = root.get("_source")
    parts: List[str] = []

    # Title and summary
    parts.append(f"عنوان: {src.get('title', '')}")
    parts.append(f"خلاصه: {src.get('summary', '')}")

    # English title
    if english := src.get("englishbody"):
        parts.append(f"عنوان انگلیسی: {english}")

    # Properties
    if "properties" in src:
        props = [
            f"{prop['Name']}: {prop['Value']}"
            for prop in src["properties"]
            if prop.get("Name") and prop.get("Value")
        ]
        parts.append("ویژگی‌ها: " + " | ".join(props))

    # Tags
    if "content_tags" in src:
        tags = [tag["Name"] for tag in src["content_tags"] if tag.get("Name")]
        parts.append("برچسب‌ها: " + "، ".join(tags))

    # Categories
    if "categories" in src:
        categories = [cat["Title"] for cat in src["categories"] if cat.get("Title")]
        parts.append("دسته‌بندی‌ها: " + "، ".join(categories))

    # Persons
    if "persons" in src:
        persons = []
        for p in src["persons"]:
            fa = p.get("PersianName", "")
            en = p.get("EnglishName", "")
            role = p.get("PersonRoleID", "")
            if fa or en:
                persons.append(f"{fa} ({en}) - نقش {role}")
        parts.append("بازیگران و عوامل فیلم: " + "، ".join(persons))
    # print (repr("\n".join(parts)))
    return "\n".join(parts)

def flatten_json(doc: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in doc.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {value}")
    return "\n".join(parts)

def flatten_json_gapfilm(doc: Dict[str, Any]) -> str:
    root = doc[0]
    src  = root.get("_source")
    parts: List[str] = []

    # Title and summary
    parts.append(f"عنوان: {src.get('title', '')}")
    parts.append(f"خلاصه: {src.get('summary', '')}")
    return "\n".join(parts)
