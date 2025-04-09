import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt


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

        history_starter = starter_model.fit(x_train, y_train,
                                            batch_size=32, epochs=500, verbose=0)

        print("Starter Model:")
        print("Starter Model Training Results:")
        train_mse_starter, train_mae_starter = starter_model.evaluate(
            x_train, y_train, verbose=2)
        print("Starter Model Test Results")
        test_mse_starter, test_mae_starter = starter_model.evaluate(
            x_test, y_test, verbose=2)

        print(history_starter.history.keys())

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

        history_model_02 = model_02.fit(x_train, y_train,
                                        batch_size=32, epochs=500, verbose=0)
        print("Model 02:")
        print("Model 02 Training Results:")
        train_mse_02, train_mae_02 = model_02.evaluate(
            x_train, y_train, verbose=2)
        print("Model 02 Test Results:")
        test_mse_02, test_mae_02 = model_02.evaluate(x_test, y_test, verbose=2)

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

        history_model_03 = model_03.fit(x_train, y_train,
                                        batch_size=32, epochs=500, verbose=0)

        print("Model 03:")
        print("Model 03 Training Results:")
        train_mse_03, train_mae_03 = model_03.evaluate(
            x_train, y_train, verbose=2)
        print("Model 03 Test Results:")
        test_mse_03, test_mae_03 = model_03.evaluate(x_test, y_test, verbose=2)

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

        history_model_04 = model_04.fit(x_train, y_train,
                                        batch_size=1024, epochs=500, verbose=0)

        print("Model 04:")
        print("Model 04 Training Results:")
        train_mse_04, train_mae_04 = model_04.evaluate(
            x_train, y_train, verbose=2)
        print("Model 04 Test Results:")
        test_mse_04, test_mae_04 = model_04.evaluate(x_test, y_test, verbose=2)

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

        history_model_05 = model_05.fit(x_train, y_train,
                                        batch_size=256, epochs=500, verbose=0)

        print("Model 05:")
        print("Model 05 Training Results:")
        train_mse_05, train_mae_05 = model_05.evaluate(
            x_train, y_train, verbose=2)
        print("Model 05 Test Results:")
        test_mse_05, test_mae_05 = model_05.evaluate(x_test, y_test, verbose=2)

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

        history_model_06 = model_06.fit(x_train, y_train,
                                        batch_size=32, epochs=500, verbose=0)
        print("Model 06:")
        print("Model 06 Training Results:")
        train_mse_06, train_mae_06 = model_06.evaluate(
            x_train, y_train, verbose=2)
        print("Model 06 Test Results:")
        test_mse_06, test_mae_06 = model_06.evaluate(x_test, y_test, verbose=2)

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
        history_model_07 = model_07.fit(x_train, y_train,
                                        batch_size=32, epochs=500, verbose=0)
        print("Model 07:")
        print("Model 07 Training Results:")
        train_mse_07, train_mae_07 = model_07.evaluate(
            x_train, y_train, verbose=2)
        print("Model 07 Test Results:")
        test_mse_07, test_mae_07 = model_07.evaluate(x_test, y_test, verbose=2)

        # MAE Plot
        plt.figure(figsize=(10, 6))
        plt.title("MAE through Epochs for All Models")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")

        plt.plot(history_starter.history['mae'], label='Starter Model')
        plt.plot(history_model_02.history['mae'], label='Model 02')
        plt.plot(history_model_03.history['mae'], label='Model 03')
        plt.plot(history_model_04.history['mae'], label='Model 04')
        plt.plot(history_model_05.history['mae'], label='Model 05')
        plt.plot(history_model_06.history['mae'], label='Model 06')
        plt.plot(history_model_07.history['mae'], label='Model 07')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # MSE Plot
        plt.figure(figsize=(10, 6))
        plt.title("MSE through Epochs for All Models")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")

        plt.plot(history_starter.history['loss'], label='Starter Model')
        plt.plot(history_model_02.history['loss'], label='Model 02')
        plt.plot(history_model_03.history['loss'], label='Model 03')
        plt.plot(history_model_04.history['loss'], label='Model 04')
        plt.plot(history_model_05.history['loss'], label='Model 05')
        plt.plot(history_model_06.history['loss'], label='Model 06')
        plt.plot(history_model_07.history['loss'], label='Model 07')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # https://github.com/user-attachments/assets/cd24221c-3b10-45e6-9e73-bff387c0d4f9
        statistics_models = np.array([
            [train_mse_starter, test_mse_starter, train_mae_starter, test_mae_starter, ((
                test_mae_starter-train_mae_starter)*100/train_mae_starter)],                       # Starter Model
            [train_mse_02, test_mse_02, train_mae_02, test_mae_02, ((
                test_mae_02-train_mae_02)*100/train_mae_02)],                                      # Model 02
            [train_mse_03, test_mse_03, train_mae_03, test_mae_03, ((
                test_mae_03-train_mae_03)*100/train_mae_03)],                                      # Model 03
            [train_mse_04, test_mse_04, train_mae_04, test_mae_04, ((
                test_mae_04-train_mae_04)*100/train_mae_04)],                                      # Model 04
            [train_mse_05, test_mse_05, train_mae_05, test_mae_05, ((
                test_mae_05-train_mae_05)*100/train_mae_05)],                                      # Model 05
            [train_mse_06, test_mse_06, train_mae_06, test_mae_06, ((
                test_mae_06-train_mae_06)*100/train_mae_06)],                                      # Model 06
            [train_mse_07, test_mse_07, train_mae_07, test_mae_07, ((
                test_mae_02-train_mae_07)*100/train_mae_07)],                                      # Model 07
        ])

        # Extract relevant columns
        test_mse = statistics_models[:, 1]
        test_mae = statistics_models[:, 3]
        variance = statistics_models[:, 4]

        # Weights
        alpha = 0.6
        beta = 0.3
        gamma = 0.1

        # Composite Score Calculation
        composite_scores = (alpha * test_mae) + (beta * variance) + (gamma * (test_mse / 100))


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
