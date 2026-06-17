from itertools import combinations

import pytest

from nonos.api._angle_parsing import _resolve_planet_file, _resolve_rotate_by


class TestParsePlanetFile:
    def test_from_filename(self):
        input_ = "test"
        assert _resolve_planet_file(planet_file=input_) == input_

    def test_from_number(self):
        input_ = 456
        assert _resolve_planet_file(planet_number=input_) == f"planet{input_}.dat"

    def test_both_args(self):
        with pytest.raises(TypeError):
            _resolve_planet_file(
                planet_file="test",
                planet_number=1,
            )

    def test_no_args(self):
        assert _resolve_planet_file() == "planet0.dat"


def mock_planet_azimuth_finder(*, planet_file: str) -> float:  # noqa: ARG001
    return 0.0


class TestParseRotationAngle:
    example_inputs = {
        "rotate_by": 1.0,
        "rotate_with": "planet0.dat",
    }
    default_kwargs = {
        "rotate_by": None,
        "rotate_with": None,
        "planet_azimuth_finder": mock_planet_azimuth_finder,
    }

    @pytest.mark.parametrize("kwargs", list(combinations(example_inputs.items(), 2)))
    def test_two_inputs(self, kwargs):
        conf = {**self.default_kwargs, **dict(kwargs)}
        with pytest.raises(
            TypeError,
            match=r"^rotate_by and rotate_with cannot be specified at the same time$",
        ):
            _resolve_rotate_by(**conf)

    def test_from_rotate_with(self):
        conf = {
            **self.default_kwargs,
            "rotate_with": self.example_inputs["rotate_with"],
        }
        result = _resolve_rotate_by(**conf)
        assert result == 0.0
