-- Dataset Comparison Table
DROP TABLE IF EXISTS dataset_comparison;

CREATE TABLE dataset_comparison (
    dataset VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    reference VARCHAR(100),
    images INTEGER,
    annotations INTEGER,
    avg_annotations_per_image FLOAT,
    avg_image_width INTEGER,
    avg_image_height INTEGER,
    avg_box_width_px INTEGER,
    avg_box_height_px INTEGER,
    acquisition_method TEXT,
    controlled_conditions VARCHAR(10),
    resolution_quality TEXT
);

INSERT INTO dataset_comparison VALUES ('LIT', 'Literature', 'Checola et al., 2024', 615, 2835, 4.61, 3218, 4939, 0, 0, 'Mixed (field, laboratory, scanned traps, smart-trap)', 'No', 'Variable');
INSERT INTO dataset_comparison VALUES ('OURS_1', 'High-Resolution', 'Current work', 1760, 2782, 1.58, 4056, 3040, 184, 179, 'Controlled imaging box with fixed geometry', 'Yes', 'High (~26.6 µm/pixel, 4056×3040)');
INSERT INTO dataset_comparison VALUES ('OURS_2', 'Field Low-Resolution', 'Current work', 3800, 13142, 3.46, 4028, 3027, 109, 106, 'Mobile phone (field conditions)', 'No', 'Variable (smartphone-dependent)');
INSERT INTO dataset_comparison VALUES ('OURS_FINAL', 'Combined (Hi-Res + Low-Res)', 'Current work', 5560, 15924, 2.86, 4036, 3031, 122, 119, 'Combined (controlled + field)', 'Mixed', 'Mixed (high + variable)');
