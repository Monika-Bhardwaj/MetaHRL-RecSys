# env/delayed_reward.py

def compute_reward(row):
    """
    KuaiRec implicit-feedback reward.
    """
    return float(row.get("watch_ratio", 0.0))