-- * - * - * - * - * - * - * - * - * -
-- SET-BASED LINKAGE (replaces the per-variable spatial_join_1pt.sql / spatial_join_2pt.sql loop)
--
-- Same rules as those templates (CHoRUS revision 894d540a), computed once per run instead of once per variable:
--   * each distinct address point is placed in its polygons once (ST_Within against a GIST index);
--   * each source table is matched to tract keys once, with the 2pt key rule
--       COALESCE(a.key, '') = COALESCE(b.key, '') OR lpad(digits(a.key), length(digits(b.key)), '0') = digits(b.key)
--     which, for tract keys that are all digits, is exactly lpad(digits(a.key), length(b.key), '0') = b.key
--     (checked below; the run stops if a polygon key is not all digits);
--   * values are averaged per tract (2pt) or taken per polygon (1pt), as before;
--   * address-years are joined to values by key and filtered with the same year rule.
-- Loaded by produce_external_exposure.sh on every run, so it also works on databases created by an older init.sql.
-- * - * - * - * - * - * - * - * - * -

-- Pure function of its arguments; IMMUTABLE lets the planner cache and inline it.
ALTER FUNCTION working.map_sdoh_year(text, integer) IMMUTABLE PARALLEL SAFE;

-- Text -> numeric exactly as the templates parse values; NULL when the text is not a plain decimal number.
CREATE OR REPLACE FUNCTION working.lk_num(t text) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT CASE WHEN NULLIF(trim(t), '') ~ '^[-+]?[0-9]+([.][0-9]+)?$' THEN NULLIF(trim(t), '')::numeric END $$;

CREATE OR REPLACE FUNCTION working.lk_digits(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT regexp_replace(COALESCE(t, ''), '[^0-9]', '', 'g') $$;

-- Integer concept id from catalog text, as the templates compute it (the default differs per column).
CREATE OR REPLACE FUNCTION working.lk_concept(t text, dflt integer) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT CASE WHEN NULLIF(t, '') ~ '^[-+]?[0-9]+([.][0-9]+)?$' THEN t::float::integer ELSE dflt END $$;

CREATE OR REPLACE FUNCTION working.lk_has_column(tbl text, col text) RETURNS boolean
LANGUAGE sql STABLE AS
$$ SELECT EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_schema = 'public' AND table_name = lower(tbl) AND lower(column_name) = lower(col)) $$;


-- 1. Address-years, distinct points and batches. Returns "<address-year rows> <points> <batches>".
CREATE OR REPLACE FUNCTION working.lk_prepare_points(batch_size integer) RETURNS text
LANGUAGE plpgsql AS
$$
DECLARE n_rows bigint; n_pts bigint; n_batches integer;
BEGIN
    DROP TABLE IF EXISTS working.lk_point, working.lk_merge;
    CREATE UNLOGGED TABLE working.lk_point AS
    SELECT row_number() OVER (ORDER BY longitude, latitude) AS pt_id, longitude, latitude,
           ST_SetSRID(ST_POINT(longitude, latitude), 4326) AS geom
    FROM (SELECT DISTINCT longitude, latitude FROM working.location_merge) d;
    ALTER TABLE working.lk_point ADD PRIMARY KEY (pt_id);

    -- lm_id fixes a total order (identical rows are interchangeable), so ids do not depend on batching.
    CREATE UNLOGGED TABLE working.lk_merge AS
    SELECT row_number() OVER (ORDER BY m.location_id, m.entity_id, m.start_date, m.end_date,
                                       m.relationship_type_concept_id, m.domain_id, m.latitude, m.longitude,
                                       m.modifier_source_value) AS lm_id,
           (dense_rank() OVER (ORDER BY m.location_id) - 1) / batch_size AS batch,
           p.pt_id,
           EXTRACT(YEAR FROM m.start_date)::integer AS yr,
           m.location_id, m.relationship_type_concept_id, m.domain_id, m.entity_id,
           m.start_date, m.end_date, m.modifier_source_value
    FROM working.location_merge m
    JOIN working.lk_point p ON p.longitude = m.longitude AND p.latitude = m.latitude;
    CREATE INDEX ON working.lk_merge (batch);
    ANALYZE working.lk_point;
    ANALYZE working.lk_merge;

    SELECT count(*), COALESCE(max(batch) + 1, 0) INTO n_rows, n_batches FROM working.lk_merge;
    SELECT count(*) INTO n_pts FROM working.lk_point;
    RETURN format('%s %s %s', n_rows, n_pts, n_batches);
END;
$$;


-- 2. Resolve every requested variable once, with the same metadata lookups and join-key fallback as the shell loop.
--    Variables the templates could not have run (missing table or column, bad dates) get a reason and produce no rows.
CREATE OR REPLACE FUNCTION working.lk_prepare_run(variables text) RETURNS integer
LANGUAGE plpgsql AS
$$
#variable_conflict use_column
DECLARE
    r record; vs record;
    geo_source text; var_source text; geo_merge text; var_merge text; var_name text; var_col text;
    v_kind text; reason text; area_tbl text; area_col text; aid integer;
    a_start date; a_end date; c_exp integer; c_src integer; c_val integer; c_unit integer;
    ident constant text := '^[A-Za-z_][A-Za-z0-9_]*$';
    n_ok integer;
BEGIN
    DROP TABLE IF EXISTS working.lk_run_var, working.lk_area;
    CREATE UNLOGGED TABLE working.lk_area (
        area_id serial PRIMARY KEY, tbl text NOT NULL, unit_col text NOT NULL, kind text NOT NULL,
        UNIQUE (tbl, unit_col, kind));
    CREATE UNLOGGED TABLE working.lk_run_var (
        var_ord integer PRIMARY KEY, var_id integer, var_name text, kind text, var_tbl text, var_merge text,
        var_col text, area_id integer, dataset_type text, attr_start date, attr_end date, attr_year integer,
        exposure_concept_id integer, exposure_source_concept_id integer, value_as_concept_id integer,
        unit_concept_id integer, status text NOT NULL);

    -- The shell loop spliced each id into SQL text, where tabs and CR/LF count as whitespace too.
    FOR r IN SELECT ord::integer AS ord, btrim(v, E' \t\r\n\f\v') AS v
             FROM unnest(string_to_array(variables, ',')) WITH ORDINALITY AS t(v, ord) LOOP
        reason := NULL; vs := NULL; area_tbl := NULL; area_col := NULL; aid := NULL;
        a_start := NULL; a_end := NULL; c_exp := NULL; c_src := NULL; c_val := NULL; c_unit := NULL;
        IF r.v !~ '^[0-9]+$' THEN
            INSERT INTO working.lk_run_var (var_ord, status) VALUES (r.ord, format('"%s" is not a variable id', r.v));
            CONTINUE;
        END IF;
        SELECT * INTO vs FROM backbone.variable_source WHERE variable_source_id = r.v::integer LIMIT 1;
        IF NOT FOUND THEN
            INSERT INTO working.lk_run_var (var_ord, var_id, status) VALUES (r.ord, r.v::integer, 'not in backbone.variable_source');
            CONTINUE;
        END IF;

        geo_source := COALESCE((SELECT split_part(download_filename, '.', 1) FROM backbone.data_source
                                WHERE data_source_uuid = vs.geom_dependency_uuid LIMIT 1), '');
        var_source := COALESCE((SELECT split_part(download_filename, '.', 1) FROM backbone.data_source
                                WHERE data_source_uuid = vs.data_source_uuid LIMIT 1), '');
        geo_merge  := COALESCE((SELECT geom_join_column FROM backbone.data_source
                                WHERE data_source_uuid = vs.geom_dependency_uuid LIMIT 1), '');
        var_merge  := COALESCE(vs.geom_join_column, '');
        var_name   := COALESCE(vs.variable_name, '');
        var_col    := COALESCE(NULLIF(vs.value_as_number_col, ''), var_name);
        IF NOT working.lk_has_column(var_source, var_merge) THEN
            IF working.lk_has_column(var_source, 'fips') THEN
                var_merge := 'FIPS';
            ELSIF working.lk_has_column(var_source, 'geoid') THEN
                var_merge := 'GEOID';
            END IF;
        END IF;
        v_kind := CASE WHEN geo_source <> var_source THEN '2pt' ELSE '1pt' END;

        IF var_source !~ ident OR to_regclass(format('public.%I', lower(var_source))) IS NULL THEN
            reason := format('source table public.%s is not loaded', var_source);
        ELSIF var_col !~ ident OR NOT working.lk_has_column(var_source, var_col) THEN
            reason := format('column %s is not in public.%s', var_col, lower(var_source));
        ELSIF length(var_name) > 50 OR position('''' IN var_name) > 0 THEN
            reason := format('variable name %s cannot be stored in exposure_source_value', var_name);
        ELSIF v_kind = '1pt' THEN
            IF NOT working.lk_has_column(var_source, 'wgs_geom') THEN
                reason := format('public.%s has no wgs_geom column', lower(var_source));
            ELSE
                area_tbl := lower(var_source);
                SELECT a.attname INTO area_col
                FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
                WHERE i.indrelid = format('public.%I', area_tbl)::regclass AND i.indisprimary AND i.indnatts = 1;
                IF area_col IS NULL THEN
                    EXECUTE format('ALTER TABLE public.%I ADD COLUMN lk_uid bigserial PRIMARY KEY', area_tbl);
                    area_col := 'lk_uid';
                END IF;
            END IF;
        ELSE
            IF geo_source !~ ident OR to_regclass(format('public.%I', lower(geo_source))) IS NULL THEN
                reason := format('geometry table public.%s is not loaded', geo_source);
            ELSIF NOT working.lk_has_column(geo_source, 'wgs_geom') THEN
                reason := format('public.%s has no wgs_geom column', lower(geo_source));
            ELSIF geo_merge !~ ident OR NOT working.lk_has_column(geo_source, geo_merge) THEN
                reason := format('join column %s is not in public.%s', geo_merge, lower(geo_source));
            ELSIF var_merge !~ ident OR NOT working.lk_has_column(var_source, var_merge) THEN
                reason := format('join column %s is not in public.%s', var_merge, lower(var_source));
            ELSE
                area_tbl := lower(geo_source);
                area_col := lower(geo_merge);
            END IF;
        END IF;

        IF reason IS NULL THEN
            BEGIN
                a_start := vs.attr_start_date::date;
                a_end   := vs.attr_end_date::date;
                c_exp   := working.lk_concept(vs.attr_concept_id, 0);
                c_src   := CASE WHEN v_kind = '2pt' THEN c_exp END;
                c_val   := working.lk_concept(vs.value_as_concept_id, NULL);
                c_unit  := working.lk_concept(vs.unit_concept_id, NULL);
            EXCEPTION WHEN others THEN
                reason := format('catalog value cannot be converted (%s)', SQLERRM);
            END;
        END IF;

        IF reason IS NULL THEN
            INSERT INTO working.lk_area (tbl, unit_col, kind) VALUES (area_tbl, area_col, v_kind)
            ON CONFLICT (tbl, unit_col, kind) DO NOTHING;
            SELECT la.area_id INTO aid FROM working.lk_area la WHERE la.tbl = area_tbl AND la.unit_col = area_col AND la.kind = v_kind;
        END IF;
        INSERT INTO working.lk_run_var VALUES (
            r.ord, vs.variable_source_id, var_name, v_kind, lower(var_source), lower(var_merge), lower(var_col), aid,
            vs.dataset_type, a_start, a_end, EXTRACT(YEAR FROM a_start)::integer,
            c_exp, c_src, c_val, c_unit, COALESCE(reason, 'ok'));
    END LOOP;

    FOR r IN SELECT var_ord, var_id, var_name, status FROM working.lk_run_var WHERE status <> 'ok' ORDER BY var_ord LOOP
        RAISE WARNING 'Variable % (%) produces no rows: %', COALESCE(r.var_id::text, '?'), COALESCE(r.var_name, ''), r.status;
    END LOOP;

    -- Freshly loaded tables have no planner statistics; without them the planner expects one row per table and
    -- picks nested loops, which brings back the quadratic cost this rewrite removes.
    -- Only the join columns are analyzed: the attribute tables are wide, and a full ANALYZE of them takes a minute.
    FOR r IN SELECT tbl, string_agg(DISTINCT quote_ident(col), ', ') AS cols
             FROM (SELECT var_tbl AS tbl, var_merge AS col FROM working.lk_run_var WHERE status = 'ok' AND kind = '2pt'
                   UNION ALL SELECT tbl, unit_col FROM working.lk_area
                   UNION ALL SELECT tbl, 'wgs_geom' FROM working.lk_area) c
             GROUP BY tbl LOOP
        EXECUTE format('ANALYZE public.%I (%s)', r.tbl, r.cols);
    END LOOP;

    -- Polygons are matched through a GIST index on wgs_geom. shp2pgsql -I only indexes geom (source SRID).
    FOR r IN SELECT DISTINCT tbl FROM working.lk_area LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_am am ON am.oid = c.relam
                       JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
                       WHERE i.indrelid = format('public.%I', r.tbl)::regclass AND am.amname = 'gist' AND a.attname = 'wgs_geom') THEN
            EXECUTE format('CREATE INDEX ON public.%I USING gist (wgs_geom)', r.tbl);
        END IF;
    END LOOP;

    ANALYZE working.lk_run_var;
    ANALYZE working.lk_area;
    DROP TABLE IF EXISTS working.lk_loc_area;
    CREATE UNLOGGED TABLE working.lk_loc_area (pt_id bigint NOT NULL, area_id integer NOT NULL, unit text);
    SELECT count(*) INTO n_ok FROM working.lk_run_var WHERE status = 'ok';
    RETURN n_ok;
END;
$$;


-- 3. Place points pt_id lo..hi in the polygons of one area. Several sessions run this on disjoint ranges.
CREATE OR REPLACE FUNCTION working.lk_assign(p_area integer, lo bigint, hi bigint) RETURNS bigint
LANGUAGE plpgsql AS
$$
DECLARE a record; n bigint;
BEGIN
    SELECT * INTO a FROM working.lk_area WHERE area_id = p_area;
    EXECUTE format(
        'INSERT INTO working.lk_loc_area (pt_id, area_id, unit)
         SELECT p.pt_id, %s, g.%I::text
         FROM working.lk_point p
         JOIN public.%I g ON public.st_within(p.geom, g.wgs_geom)
         WHERE p.pt_id BETWEEN %s AND %s', p_area, a.unit_col, a.tbl, lo, hi);
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END;
$$;


-- 4. Values for the polygons the run touches, one pass per source table; then the year filter per variable.
CREATE OR REPLACE FUNCTION working.lk_build_values() RETURNS bigint
LANGUAGE plpgsql AS
$$
DECLARE
    a record; g record; len integer; bad bigint; dup bigint; n bigint;
    avgs text; vals text; cols text;
BEGIN
    -- Every join below is an equality join; keep it a hash join even if an estimate is off.
    PERFORM set_config('enable_nestloop', 'off', true);
    CREATE INDEX ON working.lk_loc_area (pt_id, area_id);
    ANALYZE working.lk_loc_area;
    DROP TABLE IF EXISTS working.lk_var_value, working.lk_var_year;
    CREATE UNLOGGED TABLE working.lk_var_value (var_id integer NOT NULL, area_id integer NOT NULL, unit text, raw text);

    -- 2pt: the shortcut equals the template's OR rule only when every touched polygon key is all digits, and its
    -- per-key grouping equals GROUP BY wgs_geom only when no two touched polygons share a geometry.
    FOR a IN SELECT * FROM working.lk_area WHERE kind = '2pt' LOOP
        SELECT count(*) FILTER (WHERE unit IS NULL OR unit !~ '^[0-9]+$') INTO bad
        FROM (SELECT DISTINCT unit FROM working.lk_loc_area WHERE area_id = a.area_id) u;
        IF bad > 0 THEN
            RAISE EXCEPTION 'public.%.% has % touched polygon keys that are not all digits; the key rule shortcut does not apply',
                a.tbl, a.unit_col, bad;
        END IF;
        EXECUTE format(
            'SELECT count(*) - count(DISTINCT g.wgs_geom) FROM public.%I g
             WHERE g.%I::text IN (SELECT unit FROM working.lk_loc_area WHERE area_id = %s)', a.tbl, a.unit_col, a.area_id)
        INTO dup;
        IF dup > 0 THEN
            RAISE EXCEPTION 'public.% has % touched polygons with identical geometry; per-key averaging would differ from GROUP BY wgs_geom',
                a.tbl, dup;
        END IF;
    END LOOP;

    FOR g IN SELECT kind, var_tbl, var_merge, area_id, array_agg(DISTINCT var_id ORDER BY var_id) AS ids
             FROM working.lk_run_var WHERE status = 'ok' GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4 LOOP
        SELECT * INTO a FROM working.lk_area WHERE area_id = g.area_id;
        SELECT string_agg(format('(%s, t.v%s)', v.var_id, v.var_id), ', ' ORDER BY v.var_id)
        INTO vals
        FROM (SELECT DISTINCT var_id FROM working.lk_run_var WHERE var_id = ANY (g.ids)) v;
        BEGIN
            IF g.kind = '2pt' THEN
                SELECT string_agg(format('AVG(working.lk_num(a.%I::text))::text AS v%s', v.var_col, v.var_id), ', ' ORDER BY v.var_id)
                INTO avgs
                FROM (SELECT DISTINCT var_id, var_col FROM working.lk_run_var WHERE var_id = ANY (g.ids)) v;
                FOR len IN SELECT DISTINCT length(unit) FROM working.lk_loc_area WHERE area_id = g.area_id LOOP
                    EXECUTE format(
                        'INSERT INTO working.lk_var_value (var_id, area_id, unit, raw)
                         SELECT v.var_id, %s, t.unit, v.raw
                         FROM (SELECT u.unit, %s
                               FROM public.%I a
                               JOIN (SELECT DISTINCT unit FROM working.lk_loc_area WHERE area_id = %s AND length(unit) = %s) u
                                 ON lpad(working.lk_digits(a.%I::text), %s, ''0'') = u.unit
                               WHERE working.lk_digits(a.%I::text) <> ''''
                               GROUP BY u.unit) t
                         CROSS JOIN LATERAL (VALUES %s) AS v(var_id, raw)',
                        g.area_id, avgs, g.var_tbl, g.area_id, len, g.var_merge, len, g.var_merge, vals);
                END LOOP;
            ELSE
                SELECT string_agg(format('a.%I::text AS v%s', v.var_col, v.var_id), ', ' ORDER BY v.var_id)
                INTO cols
                FROM (SELECT DISTINCT var_id, var_col FROM working.lk_run_var WHERE var_id = ANY (g.ids)) v;
                EXECUTE format(
                    'INSERT INTO working.lk_var_value (var_id, area_id, unit, raw)
                     SELECT v.var_id, %s, t.unit, v.raw
                     FROM (SELECT a.%I::text AS unit, %s
                           FROM public.%I a
                           JOIN (SELECT DISTINCT unit FROM working.lk_loc_area WHERE area_id = %s) u ON a.%I::text = u.unit) t
                     CROSS JOIN LATERAL (VALUES %s) AS v(var_id, raw)',
                    g.area_id, a.unit_col, cols, g.var_tbl, g.area_id, a.unit_col, vals);
            END IF;
        EXCEPTION WHEN others THEN
            UPDATE working.lk_run_var SET status = format('values could not be read (%s)', SQLERRM) WHERE var_id = ANY (g.ids);
            RAISE WARNING 'Variables % from public.% produce no rows: %', g.ids, g.var_tbl, SQLERRM;
        END;
    END LOOP;
    -- A variable reads one area, so (var_id, unit) identifies a value.
    CREATE INDEX ON working.lk_var_value (var_id, unit);
    ANALYZE working.lk_var_value;

    -- Which variables can match an address-year at all. For mapped dataset types the year rule depends only on
    -- the calendar year; unmapped types keep every year and are decided by the full rule in link_batch.sql.
    CREATE UNLOGGED TABLE working.lk_var_year AS
    SELECT rv.var_ord, y.yr
    FROM working.lk_run_var rv
    CROSS JOIN (SELECT DISTINCT yr FROM working.lk_merge) y
    WHERE rv.status = 'ok'
      AND (working.map_sdoh_year(rv.dataset_type, y.yr) IS NULL OR working.map_sdoh_year(rv.dataset_type, y.yr) = rv.attr_year);
    ANALYZE working.lk_var_year;

    SELECT count(*) INTO n FROM working.lk_var_value;
    RETURN n;
END;
$$;
