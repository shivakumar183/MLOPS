SELECT
    r.run_uuid,
    t.value AS model_type,
    MAX(CASE WHEN m.key = 'valid_rmse' THEN m.value END) AS valid_rmse,
    MAX(CASE WHEN m.key = 'valid_mae' THEN m.value END) AS valid_mae,
    MAX(CASE WHEN m.key = 'valid_r2' THEN m.value END) AS valid_r2,
    MAX(CASE WHEN m.key = 'test_rmse' THEN m.value END) AS test_rmse,
    MAX(CASE WHEN m.key = 'test_mae' THEN m.value END) AS test_mae,
    MAX(CASE WHEN m.key = 'test_r2' THEN m.value END) AS test_r2
FROM latest_metrics m
JOIN runs r ON r.run_uuid = m.run_uuid
LEFT JOIN tags t ON t.run_uuid = r.run_uuid AND t.key = 'model_type'
GROUP BY r.run_uuid, t.value
ORDER BY model_type, r.run_uuid;
