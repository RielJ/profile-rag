-- Runs once, when the Postgres container starts with an empty volume.
-- Creates a separate database for automated tests so they never touch
-- the data you index for real use.
CREATE DATABASE profile_rag_test OWNER profile_rag;
