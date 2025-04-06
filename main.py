import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Main:
    @staticmethod
    def main():

        # Load raw data
        data = np.genfromtxt('bike_sharing_demand.csv', delimiter=',', dtype=str, skip_header=1)

        # Get all columns
        datetime = data[:, 0]
        season = data[:, 1]
        holiday = data[:, 2]
        workingday = data[:, 3]
        weather = data[:, 4]
        temp = data[:, 5]
        atemp = data[:, 6]
        humidity = data[:, 7]
        windspeed = data[:, 8]
        # casual + registered is equal to our target so these are not gonna be used
        casual = data[:, 9]              # not used
        registered = data[:, 10]         # not used
        count = data[:, 11]              # target

        hour, day, month, year = extract_datetime_features(datetime)

        print(season.shape)
        print(hour.shape)
        print(day.shape)
        print(month.shape)
        print(year.shape)

        X = np.stack([hour, day, month, year, season, holiday, workingday, weather, temp, atemp, humidity, windspeed], axis=1)
        

        # Convert to float32 to reduce computation cost
        X = X.astype("float32")
        Y = count.astype("float32")

        print(X.shape)
        print(Y.shape)

        # Normalize features using Min-Max Scaling
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)        

        # First, split into training (70%) and temp (30%) which will be used for dev + test
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)   
        


# Convert datetime string to hour, day, month, year
def extract_datetime_features(datetime_column):
    hours = []
    days = []
    months = []
    years = []
    for dt in datetime_column:
        date_part, time_part = dt.split()
        year, month, day = map(int, date_part.split('-'))
        hour = int(time_part.split(':')[0])
        hours.append(hour)
        days.append(day)
        months.append(month)
        years.append(year)
    return np.array(hours), np.array(days), np.array(months), np.array(years)

if __name__ == "__main__":
    Main.main()

