from agent1 import plan_day
from data_sources import get_user_tasks, get_device_data, get_preferences

def main():
    tasks = get_user_tasks()
    df_d = get_device_data()
    df_p = get_preferences()

    plan = plan_day(tasks, df_d, df_p)
    print(plan)

if __name__ == "__main__":
    main()