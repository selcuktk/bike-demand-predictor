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
                layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(8, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        starter_model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['mae'],
        )
        starter_model.fit(x_train, y_train,
                          batch_size=32, epochs=500, verbose=0)

        print("Starter Model:")
        print("Starter Model Training Results:")
        starter_model.evaluate(x_train, y_train, verbose=2)
        print("Starter Model Test Results")
        starter_model.evaluate(x_test, y_test, verbose=2)

        model_02 = keras.Sequential(
            [
                layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(8, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        model_02.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['mae'],
        )
        model_02.fit(x_train, y_train,
                     batch_size=32, epochs=500, verbose=0)
        print("Model 02:")
        print("Model 02 Training Results:")
        model_02.evaluate(x_train, y_train, verbose=2)
        print("Model 02 Test Results:")
        model_02.evaluate(x_test, y_test, verbose=2)

        model_03 = keras.Sequential(
            [
                layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(8, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        model_03.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.005),
            metrics=['mae'],
        )
        model_03.fit(x_train, y_train,
                     batch_size=32, epochs=500, verbose=0)

        print("Model 03:")
        print("Model 03 Training Results:")
        model_03.evaluate(x_train, y_train, verbose=2)
        print("Model 03 Test Results:")
        model_03.evaluate(x_test, y_test, verbose=2)

        model_04 = keras.Sequential(
            [
                layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(8, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        model_04.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['mae'],
        )
        model_04.fit(x_train, y_train,
                     batch_size=1024, epochs=500, verbose=0)

        print("Model 04:")
        print("Model 04 Training Results:")
        model_04.evaluate(x_train, y_train, verbose=2)
        print("Model 04 Test Results:")
        model_04.evaluate(x_test, y_test, verbose=2)

        model_05 = keras.Sequential(
            [
                layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(8, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        model_05.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['mae'],
        )
        model_05.fit(x_train, y_train,
                     batch_size=256, epochs=500, verbose=0)

        print("Model 05:")
        print("Model 05 Training Results:")
        model_05.evaluate(x_train, y_train, verbose=2)
        print("Model 05 Test Results:")
        model_05.evaluate(x_test, y_test, verbose=2)

        model_06 = keras.Sequential(
            [
                layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(128, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(16, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(4, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        model_06.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['mae'],
        )
        model_06.fit(x_train, y_train,
                     batch_size=32, epochs=500, verbose=0)
        print("Model 06:")
        print("Model 06 Training Results:")
        model_06.evaluate(x_train, y_train, verbose=2)
        print("Model 06 Test Results:")
        model_06.evaluate(x_test, y_test, verbose=2)

        model_07 = keras.Sequential(
            [
                layers.Dense(1024, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),

                layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(16, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),

                layers.Dense(1)
            ]
        )

        model_07.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['mae'],
        )
        model_07.fit(x_train, y_train,
                     batch_size=32, epochs=500, verbose=0)
        print("Model 07:")
        print("Model 07 Training Results:")
        model_07.evaluate(x_train, y_train, verbose=2)
        print("Model 07 Test Results:")
        model_07.evaluate(x_test, y_test, verbose=2)


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
