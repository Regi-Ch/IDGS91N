CREATE TABLE Dim_Country (
    id_country INT PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    region VARCHAR(100)
);

CREATE TABLE Dim_Citizenship (
    id_citizenship INT PRIMARY KEY,
    citizenship_name VARCHAR(100) NOT NULL
);

CREATE TABLE Dim_Visa (
    id_visa INT PRIMARY KEY,
    visa_type VARCHAR(100) NOT NULL
);

CREATE TABLE Fact_Migration (
    id_fact INT PRIMARY KEY,
    id_country INT,
    id_citizenship INT,
    id_visa INT,
    year INT,
    total_migrants INT,
    FOREIGN KEY (id_country) REFERENCES Dim_Country(id_country),
    FOREIGN KEY (id_citizenship) REFERENCES Dim_Citizenship(id_citizenship),
    FOREIGN KEY (id_visa) REFERENCES Dim_Visa(id_visa)
);


