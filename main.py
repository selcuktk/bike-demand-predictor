import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow import keras


class Main:
    @staticmethod
    def main():

        # Load raw data
        data = np.genfromtxt('bike_sharing_demand.csv',
                             delimiter=',', dtype=str, skip_header=1)

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

        X = np.stack([hour, day, month, year, season, holiday, workingday,
                     weather, temp, atemp, humidity, windspeed], axis=1)

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

        # Sequential API (Very convenient, not very flexible)
        starter_model = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),

                layers.Dense(256, activation='relu'),

                layers.Dense(32, activation='relu'),

                layers.Dense(8, activation='relu'),

                layers.Dense(1),
            ]
        )

        starter_model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['mae'],
        )
        starter_model.fit(x_train, y_train,
                          batch_size=32, epochs=500, verbose=2)

        starter_model.evaluate(x_test, y_test, verbose=2)

        starter_model.summary()

        mean_true = np.mean(y_train)
        mean_true2 = np.mean(y_test)

        print(str(mean_true) + ": average value of target for TRAIN examples")
        print(str(mean_true2) + ": average value of target for TEST examples")


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
