-- 1) Dimensiones
CREATE TABLE dim_time (
  time_id INTEGER PRIMARY KEY AUTO_INCREMENT,
  year INTEGER NOT NULL,
  month INTEGER,
  quarter INTEGER,
  period_label VARCHAR(20)
);

CREATE TABLE dim_country (
  country_id INTEGER PRIMARY KEY AUTO_INCREMENT,
  country_name VARCHAR(200) NOT NULL,
  iso2 CHAR(2),
  iso3 CHAR(3),
  region VARCHAR(100),
  subregion VARCHAR(100)
);

CREATE TABLE dim_visa_type (
  visa_type_id INTEGER PRIMARY KEY AUTO_INCREMENT,
  visa_type_name VARCHAR(150) NOT NULL,
  visa_category VARCHAR(100)
);

CREATE TABLE dim_last_residence (
  last_residence_id INTEGER PRIMARY KEY AUTO_INCREMENT,
  country_name VARCHAR(200) NOT NULL,
  iso3 CHAR(3),
  notes TEXT
);

-- 2) Tabla de hechos
CREATE TABLE fact_migration (
  fact_id INTEGER PRIMARY KEY AUTO_INCREMENT,
  time_id INTEGER NOT NULL,
  citizenship_country_id INTEGER NOT NULL,
  visa_type_id INTEGER NOT NULL,
  last_residence_id INTEGER NOT NULL,
  count_people BIGINT NOT NULL,
  source_file VARCHAR(255),
  ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  row_hash VARCHAR(64),
  
  FOREIGN KEY (time_id) REFERENCES dim_time(time_id),
  FOREIGN KEY (citizenship_country_id) REFERENCES dim_country(country_id),
  FOREIGN KEY (visa_type_id) REFERENCES dim_visa_type(visa_type_id),
  FOREIGN KEY (last_residence_id) REFERENCES dim_last_residence(last_residence_id)
);

-- Índices para consultas rápidas
CREATE INDEX idx_fact_time ON fact_migration(time_id);
CREATE INDEX idx_fact_citizenship ON fact_migration(citizenship_country_id);
CREATE INDEX idx_fact_visa ON fact_migration(visa_type_id);