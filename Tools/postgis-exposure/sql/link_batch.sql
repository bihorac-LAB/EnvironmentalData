-- EXTERNAL_EXPOSURE rows for one batch of locations (psql variable :batch), written to stdout as CSV.
-- Columns and rules follow spatial_join_1pt.sql / spatial_join_2pt.sql; var_ord, lm_id and unit are sort keys
-- for the final id assignment (see export in produce_external_exposure.sh).
SET work_mem = '128MB';
COPY (
SELECT rv.var_ord
     , lm.lm_id
     , la.unit
     , lm.location_id
     , lm.entity_id
     , rv.exposure_concept_id
     , lm.start_date
     , lm.start_date::timestamp
     , lm.end_date
     , lm.end_date::timestamp
     , 0
     , 0
     , rv.exposure_source_concept_id
     , rv.var_name
     , CAST(NULL AS VARCHAR(50))
     , CAST(NULL AS VARCHAR(50))
     , CAST(NULL AS INTEGER)
     , LEFT(lm.modifier_source_value, 50)
     , CAST(NULL AS INTEGER)
     , working.lk_num(vv.raw)::float
     , rv.value_as_concept_id
     , rv.unit_concept_id
     , working.map_sdoh_year(rv.dataset_type, lm.yr)
     , CASE
           WHEN working.map_sdoh_year(rv.dataset_type, lm.yr) IS NULL THEN 'not_applicable'
           WHEN working.map_sdoh_year(rv.dataset_type, lm.yr) = lm.yr THEN 'exact_year'
           ELSE 'nearest_year'
       END
FROM working.lk_merge lm
JOIN working.lk_var_year vy ON vy.yr = lm.yr
JOIN working.lk_run_var rv ON rv.var_ord = vy.var_ord
JOIN working.lk_loc_area la ON la.pt_id = lm.pt_id AND la.area_id = rv.area_id
JOIN working.lk_var_value vv ON vv.var_id = rv.var_id AND vv.unit = la.unit
WHERE lm.batch = :batch
  AND rv.status = 'ok'
  AND (
        (
             working.map_sdoh_year(rv.dataset_type, COALESCE(lm.yr, EXTRACT(YEAR FROM lm.end_date)::integer)) IS NOT NULL
             AND working.map_sdoh_year(rv.dataset_type, COALESCE(lm.yr, EXTRACT(YEAR FROM lm.end_date)::integer)) = rv.attr_year
        )
        OR
        (
             working.map_sdoh_year(rv.dataset_type, COALESCE(lm.yr, EXTRACT(YEAR FROM lm.end_date)::integer)) IS NULL
             AND (
                  lm.start_date BETWEEN rv.attr_start AND rv.attr_end
                  OR lm.end_date BETWEEN rv.attr_start AND rv.attr_end
                  OR (lm.start_date <= rv.attr_start AND lm.end_date >= rv.attr_end)
             )
        )
  )
) TO STDOUT (FORMAT csv);
