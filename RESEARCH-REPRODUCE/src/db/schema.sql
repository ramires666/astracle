-- DB schema for ostrofun
-- All tables use subject_id

-- Subjects table
CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    birth_dt_utc TIMESTAMPTZ NOT NULL,
    birth_lat DOUBLE PRECISION,
    birth_lon DOUBLE PRECISION
);

-- Market data (close only)
CREATE TABLE IF NOT EXISTS market_daily (
    subject_id TEXT NOT NULL,
    date DATE NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (subject_id, date),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Body positions
CREATE TABLE IF NOT EXISTS astro_bodies_daily (
    subject_id TEXT NOT NULL,
    date DATE NOT NULL,
    body TEXT NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    speed DOUBLE PRECISION NOT NULL,
    is_retro BOOLEAN NOT NULL,
    sign TEXT NOT NULL,
    declination DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (subject_id, date, body),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Aspects between bodies
CREATE TABLE IF NOT EXISTS astro_aspects_daily (
    subject_id TEXT NOT NULL,
    date DATE NOT NULL,
    p1 TEXT NOT NULL,
    p2 TEXT NOT NULL,
    aspect TEXT NOT NULL,
    orb DOUBLE PRECISION NOT NULL,
    is_exact BOOLEAN NOT NULL,
    is_applying BOOLEAN NOT NULL,
    PRIMARY KEY (subject_id, date, p1, p2, aspect),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Natal bodies
CREATE TABLE IF NOT EXISTS natal_bodies (
    subject_id TEXT NOT NULL,
    body TEXT NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    speed DOUBLE PRECISION NOT NULL,
    is_retro BOOLEAN NOT NULL,
    sign TEXT NOT NULL,
    declination DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (subject_id, body),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Natal aspects
CREATE TABLE IF NOT EXISTS natal_aspects (
    subject_id TEXT NOT NULL,
    p1 TEXT NOT NULL,
    p2 TEXT NOT NULL,
    aspect TEXT NOT NULL,
    orb DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (subject_id, p1, p2, aspect),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Transit aspects
CREATE TABLE IF NOT EXISTS transit_aspects_daily (
    subject_id TEXT NOT NULL,
    date DATE NOT NULL,
    transit_body TEXT NOT NULL,
    natal_body TEXT NOT NULL,
    aspect TEXT NOT NULL,
    orb DOUBLE PRECISION NOT NULL,
    is_exact BOOLEAN NOT NULL,
    is_applying BOOLEAN NOT NULL,
    PRIMARY KEY (subject_id, date, transit_body, natal_body, aspect),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Wide features table
CREATE TABLE IF NOT EXISTS features_daily (
    subject_id TEXT NOT NULL,
    date DATE NOT NULL,
    -- feature columns are added by a separate migration
    PRIMARY KEY (subject_id, date),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Oracle labels
CREATE TABLE IF NOT EXISTS labels_daily (
    subject_id TEXT NOT NULL,
    date DATE NOT NULL,
    target INTEGER NOT NULL,
    smoothed_close DOUBLE PRECISION,
    smooth_slope DOUBLE PRECISION,
    PRIMARY KEY (subject_id, date),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- --------------------------
-- TimescaleDB (optional)
-- --------------------------
-- CREATE EXTENSION IF NOT EXISTS timescaledb;
-- SELECT create_hypertable('market_daily', 'date', if_not_exists => TRUE);
-- SELECT create_hypertable('astro_bodies_daily', 'date', if_not_exists => TRUE);
-- SELECT create_hypertable('astro_aspects_daily', 'date', if_not_exists => TRUE);
-- SELECT create_hypertable('transit_aspects_daily', 'date', if_not_exists => TRUE);
-- SELECT create_hypertable('features_daily', 'date', if_not_exists => TRUE);
-- SELECT create_hypertable('labels_daily', 'date', if_not_exists => TRUE);
