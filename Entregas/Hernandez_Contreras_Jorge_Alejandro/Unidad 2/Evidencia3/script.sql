-- Dimensiones
CREATE TABLE dim_date (
    date_key DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(20),
    quarter INTEGER
);

CREATE TABLE dim_passenger_type (
    passenger_type_key SERIAL PRIMARY KEY,
    passenger_type VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE dim_direction (
    direction_key SERIAL PRIMARY KEY,
    direction VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE dim_citizenship (
    citizenship_key SERIAL PRIMARY KEY,
    citizenship VARCHAR(150) UNIQUE NOT NULL,
    iso_code VARCHAR(10)
);

CREATE TABLE dim_visa (
    visa_key SERIAL PRIMARY KEY,
    visa_type VARCHAR(150) UNIQUE NOT NULL
);

CREATE TABLE dim_country_residence (
    country_residence_key SERIAL PRIMARY KEY,
    country_name VARCHAR(150) UNIQUE NOT NULL,
    iso_code VARCHAR(10),
    region VARCHAR(100)
);

CREATE TABLE dim_status (
    status_key SERIAL PRIMARY KEY,
    status VARCHAR(50) UNIQUE NOT NULL
);

-- Tabla de hechos
CREATE TABLE fact_migration (
    fact_id SERIAL PRIMARY KEY,
    date_key DATE NOT NULL,
    passenger_type_key INTEGER NOT NULL,
    direction_key INTEGER NOT NULL,
    citizenship_key INTEGER NOT NULL,
    visa_key INTEGER NOT NULL,
    country_residence_key INTEGER NOT NULL,
    status_key INTEGER NOT NULL,
    estimate BIGINT NOT NULL,
    standard_error BIGINT,
    CONSTRAINT fk_date FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    CONSTRAINT fk_passenger FOREIGN KEY (passenger_type_key) REFERENCES dim_passenger_type(passenger_type_key),
    CONSTRAINT fk_direction FOREIGN KEY (direction_key) REFERENCES dim_direction(direction_key),
    CONSTRAINT fk_citizenship FOREIGN KEY (citizenship_key) REFERENCES dim_citizenship(citizenship_key),
    CONSTRAINT fk_visa FOREIGN KEY (visa_key) REFERENCES dim_visa(visa_key),
    CONSTRAINT fk_countryres FOREIGN KEY (country_residence_key) REFERENCES dim_country_residence(country_residence_key),
    CONSTRAINT fk_status FOREIGN KEY (status_key) REFERENCES dim_status(status_key)
);
