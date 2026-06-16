"""Turn the cleaned frame into model-ready features.

Skills arrive as one comma-separated string per row. We expand that into a
binary column per known skill plus a count of how many skills the row lists.
"""


def add_skill_flags(df, skills):
    df = df.copy()
    lowered = df["Skills"].astype(str).str.lower()
    for skill in skills:
        df[skill] = lowered.str.contains(skill.lower(), regex=False).astype(int)
    return df


def add_skill_count(df):
    df = df.copy()
    df["skill_count"] = df["Skills"].astype(str).apply(
        lambda x: len([part for part in x.split(",") if part.strip()])
    )
    return df


def build_features(df, skills):
    """Add skill flags + skill_count, then drop the raw Skills string."""
    df = add_skill_flags(df, skills)
    df = add_skill_count(df)
    return df.drop(columns=["Skills"])
