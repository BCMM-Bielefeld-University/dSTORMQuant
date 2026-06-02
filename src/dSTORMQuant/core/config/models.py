from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Required keys for required_metadata_columns dict (order does not matter)
REQUIRED_METADATA_KEYS = frozenset(
    {
        "file_name",
        "first_channel_index",
        "first_ch_frame_last",
        "second_channel_index",
        "second_ch_frame_last",
    }
)

# Optional keys for optional_metadata_columns (Excel columns may be absent)
OPTIONAL_METADATA_KEYS = frozenset(
    {
        "experiment_number",
        "initials",
        "tag",
        "first_channel_label",
        "second_channel_label",
    }
)


class InputDataConfig(BaseModel):
    """Input data configuration."""

    file_format: Literal["csv"] = Field(
        default="csv", description="Format of input files"
    )
    required_columns: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Localization CSV columns validated at load (see docs). "
            "Minimum is typically x (nm), y (nm), channelIndex, frameIndex; "
            "quality metrics may be optional and filled for drift conversion when absent."
        ),
    )
    xlsx_filename: str = Field(..., description="Metadata Excel filename")
    skip_frames: int = Field(
        200,
        ge=0,
        description="Frames to skip at the start of each channel's frame range",
    )
    required_metadata_columns: dict[str, str] = Field(
        ...,
        description=(
            "Metadata Excel column names by role. Required keys: file_name, "
            "first_channel_index, first_ch_frame_last, second_channel_index, "
            "second_ch_frame_last"
        ),
    )
    optional_metadata_columns: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional metadata Excel columns: experiment_number, initials, tag, "
            "first_channel_label, second_channel_label (values are Excel header names)"
        ),
    )

    @field_validator("required_columns")
    @classmethod
    def validate_required_columns(cls, v):
        """Ensure all required column names are non-empty strings.

        Args:
            cls: Validator class (unused).
            v: ``required_columns`` value from the model.

        Returns:
            Validated list of column names.

        Raises:
            ValueError: If any entry is not a non-empty string.
        """
        if not all(isinstance(c, str) and c for c in v):
            raise ValueError(
                "required_columns must be a non-empty list of non-empty strings"
            )
        return v

    @field_validator("required_metadata_columns")
    @classmethod
    def validate_required_metadata_columns(cls, v):
        """Ensure required metadata column mapping is complete and valid.

        Args:
            cls: Validator class (unused).
            v: ``required_metadata_columns`` dict from the model.

        Returns:
            Validated metadata column mapping.

        Raises:
            ValueError: If keys differ from ``REQUIRED_METADATA_KEYS`` or values
                are not non-empty strings.
        """
        if not isinstance(v, dict):
            raise ValueError("required_metadata_columns must be a dict")
        keys = set(v.keys())
        if keys != REQUIRED_METADATA_KEYS:
            missing = REQUIRED_METADATA_KEYS - keys
            extra = keys - REQUIRED_METADATA_KEYS
            msg = []
            if missing:
                msg.append(f"missing keys: {sorted(missing)}")
            if extra:
                msg.append(f"unknown keys: {sorted(extra)}")
            raise ValueError(
                f"required_metadata_columns must have exactly these keys: {sorted(REQUIRED_METADATA_KEYS)}. {'; '.join(msg)}"
            )
        if not all(isinstance(c, str) and c for c in v.values()):
            raise ValueError(
                "required_metadata_columns values must be non-empty strings"
            )
        return v

    @field_validator("optional_metadata_columns")
    @classmethod
    def validate_optional_metadata_columns(cls, v):
        """Ensure optional metadata keys are allowed and values are valid strings.

        Args:
            cls: Validator class (unused).
            v: ``optional_metadata_columns`` dict from the model.

        Returns:
            Validated optional metadata column mapping.

        Raises:
            ValueError: If unknown keys are present or values are not non-empty strings.
        """
        if not isinstance(v, dict):
            raise ValueError("optional_metadata_columns must be a dict")
        keys = set(v.keys())
        if not keys.issubset(OPTIONAL_METADATA_KEYS):
            extra = keys - OPTIONAL_METADATA_KEYS
            raise ValueError(
                f"optional_metadata_columns has unknown keys: {sorted(extra)}. "
                f"Allowed: {sorted(OPTIONAL_METADATA_KEYS)}"
            )
        if not all(isinstance(c, str) and c for c in v.values()):
            raise ValueError(
                "optional_metadata_columns values must be non-empty strings when set"
            )
        return v


class DataConfig(BaseModel):
    """Data input/output configuration."""

    input: InputDataConfig


class ChannelConfig(BaseModel):
    """Individual channel configuration."""

    color: str = Field(..., min_length=1, description="Hex color code for the channel")

    @field_validator("color")
    @classmethod
    def validate_color(cls, v):
        """Validate channel color is a ``#RRGGBB`` hex string.

        Args:
            cls: Validator class (unused).
            v: Color string from the model.

        Returns:
            Validated color string.

        Raises:
            ValueError: If the value is empty or not seven-character hex format.
        """
        if not v or not isinstance(v, str):
            raise ValueError("Color must be a non-empty string hex color")
        if not v.startswith("#") or len(v) != 7:
            raise ValueError(f"Color must be in hex format #RRGGBB, got: {v}")
        return v


class ChannelsConfig(BaseModel):
    """Channels configuration."""

    Ch1: ChannelConfig
    Ch2: ChannelConfig


class IntensityFilterConfig(BaseModel):
    """Intensity filtering parameters (minimum photon count)."""

    min_value: float = Field(
        ..., gt=0, description="Minimum photon count required (filters out dim spots)"
    )


class LocalizationPrecisionFilter(BaseModel):
    """Localization precision filtering parameters."""

    threshold_value: float = Field(
        ..., gt=0, description="Maximum localization precision (nm)"
    )


class PValueFilter(BaseModel):
    """P-value filtering parameters."""

    threshold_value: float = Field(..., gt=0, le=1, description="Maximum p-value")


class SigmaFilter(BaseModel):
    """Sigma filtering parameters."""

    min_value: float = Field(..., ge=0, description="Minimum sigma (nm)")
    max_value: float = Field(..., gt=0, description="Maximum sigma (nm)")

    @field_validator("max_value")
    @classmethod
    def max_greater_than_min(cls, v, info):
        """Ensure ``max_value`` is greater than ``min_value``.

        Args:
            cls: Validator class (unused).
            v: ``max_value`` from the model.
            info: Pydantic validation context with sibling field values.

        Returns:
            Validated ``max_value``.

        Raises:
            ValueError: If ``max_value <= min_value``.
        """
        if "min_value" in info.data and v <= info.data["min_value"]:
            raise ValueError(
                f"max_value ({v}) must be greater than min_value ({info.data['min_value']})"
            )
        return v


class FilteringConfig(BaseModel):
    """Filtering parameters configuration."""

    intensity: IntensityFilterConfig
    localization_precision: LocalizationPrecisionFilter
    p_value: PValueFilter
    sigma: SigmaFilter


class DriftValidationConfig(BaseModel):
    """Drift validation parameters."""

    use: bool = Field(..., description="Enable drift validation")
    max_segment_drift_nm: float = Field(
        ..., gt=0, description="Maximum allowed drift per segment in nm"
    )
    n_previous_segments: int = Field(
        default=10,
        gt=0,
        description="Number of previous segments to average for replacement",
    )


class SanityChecksConfig(BaseModel):
    """Sanity check parameters after drift correction."""
    # This option is no longer provided in config files; keep default True in code
    remove_invalid_values: bool = Field(
        default=True, description="Remove inf/nan values (forced True if absent)"
    )
    filter_boundary: bool = Field(..., description="Remove x<=0 or y<=0")
    filter_parameters: bool = Field(..., description="Remove sx<=0, sy<=0, or lp<=0")


class DriftCorrectionConfig(BaseModel):
    """Drift correction parameters."""

    pixel_size: int = Field(..., gt=0, description="Pixel size in nm")
    segmentation: int = Field(..., gt=0, description="Number of segments")
    intersect_d_nm: float = Field(
        ..., gt=0, description="Intersection distance in nanometers"
    )
    roi_r_nm: float = Field(..., gt=0, description="ROI radius in nanometers")
    drift_validation: DriftValidationConfig = Field(
        ..., description="Drift validation settings"
    )
    sanity_checks: SanityChecksConfig = Field(..., description="Sanity check filters")


class TemporalGroupingConfig(BaseModel):
    """Temporal grouping configuration."""

    use: bool = Field(..., description="Enable/disable temporal grouping")
    max_frame_gap: int = Field(..., ge=0, description="Maximum frame gap")
    max_distance_nm: float = Field(..., gt=0, description="Maximum distance in nm")
    min_duration: int = Field(..., ge=1, description="Minimum duration in frames")
    max_duration: int = Field(..., ge=1, description="Maximum duration in frames")

    @field_validator("max_duration")
    @classmethod
    def max_duration_ge_min_duration(cls, v, info):
        """Ensure ``max_duration`` is not less than ``min_duration``.

        Args:
            cls: Validator class (unused).
            v: ``max_duration`` from the model.
            info: Pydantic validation context with sibling field values.

        Returns:
            Validated ``max_duration``.

        Raises:
            ValueError: If ``max_duration < min_duration``.
        """
        if "min_duration" in info.data and v < info.data["min_duration"]:
            raise ValueError(
                f"max_duration ({v}) must be >= min_duration ({info.data['min_duration']})"
            )
        return v


class GridCellDetection(BaseModel):
    """Grid-based cell detection parameters."""

    grid_size: int = Field(..., gt=0, description="Grid size for cell detection")
    high_density_threshold: int = Field(..., ge=1, description="High density threshold")


class VornoiCellDetection(BaseModel):
    """Voronoi cell detection parameters."""

    percentile: float = Field(..., ge=1, le=100, description="Percentile for clipping")


class CellDetectionConfig(BaseModel):
    """Cell detection configuration."""

    use: bool = Field(..., description="Enable/disable cell detection")
    approach: Literal["vornoi", "grid"] = Field(
        ..., description="Cell detection approach"
    )
    grid: GridCellDetection
    vornoi: VornoiCellDetection


class NearestNeighborConfig(BaseModel):
    """Nearest neighbor analysis configuration (kNN only)."""

    use: bool = Field(..., description="Enable/disable nearest neighbor analysis")
    radius: float = Field(
        ..., gt=0, description="Radius for mean distance within radius (nm)"
    )
    k: int = Field(..., ge=1, description="Number of nearest neighbors")
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = Field(
        ..., description="kNN algorithm"
    )
    metric: str = Field(..., description="Distance metric")


class DBSCANParams(BaseModel):
    """DBSCAN clustering parameters."""

    eps: float = Field(..., gt=0, description="Maximum distance between samples (nm)")
    min_samples: int = Field(..., ge=1, description="Minimum samples for core point")


class HDBSCANParams(BaseModel):
    """HDBSCAN clustering parameters."""

    min_cluster_size: int = Field(..., ge=2, description="Minimum cluster size")
    min_samples: int = Field(..., ge=1, description="Minimum samples for core point")


class FinderParams(BaseModel):
    """FINDER clustering parameters (unbiased parameter selection)."""

    threshold: int = Field(
        10, ge=1, description="Default minPts (minimum points for cluster)"
    )
    points_per_dimension: int = Field(
        15, ge=2, description="Grid resolution for parameter search"
    )
    algorithm: Literal["dbscan", "DbscanLoop"] = Field(
        "dbscan", description="Inner algorithm"
    )
    min_threshold: int = Field(5, ge=1, description="Minimum threshold to search")
    max_threshold: int = Field(21, ge=1, description="Maximum threshold to search")
    decay: float = Field(0.5, ge=0, le=1, description="Decay for parameter selection")


class ChannelClusteringConfig(BaseModel):
    """Per-channel clustering configuration."""

    use: bool = Field(
        default=True,
        description="Run clustering for this channel (false keeps cluster=-1 for that channel)",
    )
    method: Literal["dbscan", "hdbscan", "finder"] = Field(
        ..., description="Clustering method"
    )
    dbscan: DBSCANParams
    hdbscan: HDBSCANParams
    finder: FinderParams = Field(
        default_factory=FinderParams,
        description="FINDER parameters (used when method is finder)",
    )


class ClusteringConfig(BaseModel):
    """Clustering configuration."""

    use: bool = Field(..., description="Enable/disable clustering")
    use_cluster_knn: bool = Field(..., description="Enable/disable KNN of clustering")
    ch1: ChannelClusteringConfig
    ch2: ChannelClusteringConfig


class CBCParams(BaseModel):
    """Coordinate-Based Colocalization parameters."""

    radius: float = Field(..., gt=0, description="Radius for CBC calculation (nm)")
    n_steps: int = Field(..., ge=1, description="Number of steps for CBC computation")


class ColocalizationConfig(BaseModel):
    """Colocalization analysis configuration."""

    use: bool = Field(..., description="Enable/disable colocalization analysis")
    cbc: CBCParams


class VisualizationConfig(BaseModel):
    """Pipeline visualization (Napari screenshots on/off)."""

    use_napari: bool = Field(
        default=True,
        description=(
            "If true, save Napari-based point previews and cluster overview images. "
            "If false, skip those outputs (headless/cloud; no substitute images)."
        ),
    )


class SMMLConfig(BaseModel):
    """Main SMLM pipeline configuration."""

    data: DataConfig
    channels: ChannelsConfig
    filtering: FilteringConfig
    drift_correction: DriftCorrectionConfig
    temporal_grouping: TemporalGroupingConfig
    cell_detection: CellDetectionConfig
    nearest_neighbor_analysis: NearestNeighborConfig
    clustering: ClusteringConfig
    colocalization: ColocalizationConfig
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Napari screenshot toggles",
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Forbid extra fields not defined in model
