from datetime import datetime, timedelta
import calendar as cal


def add_day(start_date):
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def add_days(start_date, days):
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    return (dt + timedelta(days=int(days))).strftime("%Y-%m-%d")


def is_leap_year(start_date):
    year = int(datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y"))
    return bool(cal.isleap(year))


# This class carries the major bodies physical parameters
class major_bodies:
    au = 149597870.7       # km
    day = 86400            # seconds
    mu = 1.32712440018e11  # km^3/s^2
    Re = 6378.14           # km


if __name__ == "__main__":
    start = "2024-01-02"
    print(f"start={start} next_day={add_day(start)} leap={is_leap_year(start)}")