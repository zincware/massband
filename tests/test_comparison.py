"""Tests for node comparison methods."""

import numpy as np
import pytest

from massband.rdf.core import RadialDistributionFunction


class TestRadialDistributionFunctionCompare:
    """Tests for RadialDistributionFunction.compare()."""

    @pytest.fixture
    def mock_rdf_nodes(self):
        """Create mock RDF nodes for testing."""
        # Create three mock nodes with similar but slightly different RDF data
        nodes = []
        for i in range(3):
            node = type("MockRDFNode", (), {})()
            node.name = f"rdf_node_{i}"
            bin_centers = np.linspace(0.5, 10.0, 100)
            # Create slightly different g(r) curves
            g_r = 1.0 + np.exp(-((bin_centers - 2.0 - i * 0.1) ** 2) / 0.5)

            node.rdf = {
                "Li-Li": {
                    "bin_centers": bin_centers.tolist(),
                    "g_r": g_r.tolist(),
                    "g_r_std": None,
                    "g_r_ensemble": None,
                    "unit": "Å",
                    "number_density_a": 0.01,
                    "number_density_b": 0.01,
                },
                "F-F": {
                    "bin_centers": bin_centers.tolist(),
                    "g_r": (g_r * 1.2).tolist(),
                    "g_r_std": None,
                    "g_r_ensemble": None,
                    "unit": "Å",
                    "number_density_a": 0.02,
                    "number_density_b": 0.02,
                },
            }
            nodes.append(node)
        return nodes

    def test_compare_requires_two_nodes(self, mock_rdf_nodes):
        """Test that compare requires at least 2 nodes."""
        with pytest.raises(ValueError, match="At least two nodes"):
            RadialDistributionFunction.compare(mock_rdf_nodes[0])

    def test_compare_basic(self, mock_rdf_nodes):
        """Test basic comparison of RDF nodes."""
        result = RadialDistributionFunction.compare(*mock_rdf_nodes)

        assert "figures" in result
        figures = result["figures"]

        # Check that overlay plots are created for each pair
        assert "overlay_Li-Li" in figures
        assert "overlay_F-F" in figures

        # Check that summary plots are created
        assert "peak_positions" in figures
        assert "peak_heights" in figures
        assert "coordination_numbers" in figures

    def test_compare_with_labels(self, mock_rdf_nodes):
        """Test comparison with custom labels."""
        labels = ["Method A", "Method B", "Method C"]
        result = RadialDistributionFunction.compare(*mock_rdf_nodes, labels=labels)

        assert "figures" in result
        assert len(result["figures"]) > 0

    def test_compare_with_pair_selection(self, mock_rdf_nodes):
        """Test comparison with specific pair selection."""
        result = RadialDistributionFunction.compare(*mock_rdf_nodes, pairs=["Li-Li"])

        figures = result["figures"]
        assert "overlay_Li-Li" in figures
        assert "overlay_F-F" not in figures

    def test_compare_mismatched_labels(self, mock_rdf_nodes):
        """Test that mismatched labels raise error."""
        with pytest.raises(ValueError, match="Number of labels"):
            RadialDistributionFunction.compare(*mock_rdf_nodes, labels=["A", "B"])

    def test_compare_no_common_pairs(self):
        """Test error when nodes have no common pairs."""
        node1 = type("MockRDFNode", (), {})()
        node1.name = "node1"
        node1.rdf = {
            "Li-Li": {
                "bin_centers": [1.0],
                "g_r": [1.0],
                "g_r_std": None,
                "g_r_ensemble": None,
                "unit": "Å",
                "number_density_a": 0.01,
                "number_density_b": 0.01,
            }
        }

        node2 = type("MockRDFNode", (), {})()
        node2.name = "node2"
        node2.rdf = {
            "F-F": {
                "bin_centers": [1.0],
                "g_r": [1.0],
                "g_r_std": None,
                "g_r_ensemble": None,
                "unit": "Å",
                "number_density_a": 0.02,
                "number_density_b": 0.02,
            }
        }

        with pytest.raises(ValueError, match="No common pairs"):
            RadialDistributionFunction.compare(node1, node2)

    def test_compare_with_uncertainties(self):
        """Test comparison with RDF uncertainties."""
        nodes = []
        for i in range(2):
            node = type("MockRDFNode", (), {})()
            node.name = f"node_with_uncert_{i}"
            bin_centers = np.linspace(0.5, 10.0, 50)
            g_r = 1.0 + np.exp(-((bin_centers - 2.0) ** 2) / 0.5)
            g_r_std = 0.1 * np.ones_like(g_r)

            node.rdf = {
                "Li-Li": {
                    "bin_centers": bin_centers.tolist(),
                    "g_r": g_r.tolist(),
                    "g_r_std": g_r_std.tolist(),
                    "g_r_ensemble": None,
                    "unit": "Å",
                    "number_density_a": 0.01,
                    "number_density_b": 0.01,
                }
            }
            nodes.append(node)

        result = RadialDistributionFunction.compare(*nodes)
        assert "figures" in result
        assert "overlay_Li-Li" in result["figures"]


class TestKinisiSelfDiffusionCompare:
    """Tests for KinisiSelfDiffusion.compare()."""

    @pytest.fixture
    def mock_diffusion_nodes(self, tmp_path):
        """Create mock diffusion nodes for testing."""

        nodes = []
        for i in range(3):
            node = type("MockDiffusionNode", (), {})()
            node.name = f"diffusion_node_{i}"
            # Create temporary data path for this node
            node.data_path = tmp_path / f"node_{i}"
            node.data_path.mkdir(parents=True, exist_ok=True)

            node.diffusion = {
                "[Li+]": {
                    "mean": 1e-5 * (1 + i * 0.1),
                    "std": 1e-6,
                    "var": 1e-12,
                    "occurrences": 10,
                    "unit": "cm^2/s",
                    "box": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                },
                "EC": {
                    "mean": 5e-6 * (1 + i * 0.05),
                    "std": 5e-7,
                    "var": 2.5e-13,
                    "occurrences": 20,
                    "unit": "cm^2/s",
                    "box": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                },
            }
            nodes.append(node)
        return nodes

    def test_compare_basic(self, mock_diffusion_nodes):
        """Test basic comparison of diffusion nodes."""
        from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion

        result = KinisiSelfDiffusion.compare(*mock_diffusion_nodes)

        assert "figures" in result
        figures = result["figures"]

        assert "diffusion_coefficients" in figures
        assert "relative_differences" in figures

    def test_compare_with_structure_selection(self, mock_diffusion_nodes):
        """Test comparison with specific structure selection."""
        from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion

        result = KinisiSelfDiffusion.compare(*mock_diffusion_nodes, structures=["[Li+]"])

        assert "figures" in result
        # Should only compare Li+, not EC
        # This is implicit - the comparison should succeed without error

    def test_compare_with_msd_data(self, tmp_path):
        """Test comparison with actual MSD data files."""
        import scipp as sc

        from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion
        from massband.utils import sanitize_structure_name

        structure_name = "[Li+]"
        safe_structure = sanitize_structure_name(structure_name)

        nodes = []
        for i in range(2):
            node = type("MockDiffusionNode", (), {})()
            node.name = f"diffusion_with_msd_{i}"
            node.data_path = tmp_path / f"node_msd_{i}"
            node.data_path.mkdir(parents=True, exist_ok=True)

            node.diffusion = {
                structure_name: {
                    "mean": 1e-5 * (1 + i * 0.1),
                    "std": 1e-6,
                    "var": 1e-12,
                    "occurrences": 10,
                    "unit": "cm^2/s",
                    "box": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                }
            }

            # Create mock MSD and dt data
            time_points = np.linspace(0, 100, 50)  # 0-100 ps
            msd_vals = time_points * (1.0 + i * 0.1)  # Linear MSD for testing

            msd_sc = sc.array(dims=["time"], values=msd_vals, unit="angstrom^2")
            dt_sc = sc.array(dims=["time"], values=time_points, unit="ps")

            # Save to HDF5 files with sanitized names
            msd_sc.save_hdf5(node.data_path / f"{safe_structure}_msd.h5")
            dt_sc.save_hdf5(node.data_path / f"{safe_structure}_dt.h5")

            nodes.append(node)

        result = KinisiSelfDiffusion.compare(*nodes)

        assert "figures" in result
        figures = result["figures"]

        # Should have diffusion coefficient and MSD plots
        assert "diffusion_coefficients" in figures
        assert "relative_differences" in figures
        assert f"msd_{structure_name}" in figures


class TestPotentialOfMeanForceCompare:
    """Tests for PotentialOfMeanForce.compare()."""

    @pytest.fixture
    def mock_pmf_nodes(self):
        """Create mock PMF nodes for testing."""
        nodes = []
        for i in range(2):
            node = type("MockPMFNode", (), {})()
            node.name = f"pmf_node_{i}"
            r = np.linspace(1.0, 10.0, 100)
            # Create a PMF with a well and barrier
            pmf = 0.5 * (r - 2.0 - i * 0.1) ** 2 - 0.3

            node.pmf = {
                "Li-F": {
                    "r": r.tolist(),
                    "pmf": pmf.tolist(),
                    "pmf_std": None,
                    "unit": "eV",
                }
            }
            nodes.append(node)
        return nodes

    def test_compare_basic(self, mock_pmf_nodes):
        """Test basic comparison of PMF nodes."""
        from massband.pmf import PotentialOfMeanForce

        result = PotentialOfMeanForce.compare(*mock_pmf_nodes)

        assert "figures" in result
        figures = result["figures"]

        assert "overlay_Li-F" in figures
        assert "barrier_heights" in figures
        assert "well_depths" in figures
        assert "minima_positions" in figures

    def test_compare_with_alignment(self, mock_pmf_nodes):
        """Test comparison with minima alignment."""
        from massband.pmf import PotentialOfMeanForce

        result = PotentialOfMeanForce.compare(*mock_pmf_nodes, align_minima=True)

        assert "figures" in result
        assert "overlay_Li-F" in result["figures"]


class TestKinisiDiffusionArrheniusCompare:
    """Tests for KinisiDiffusionArrhenius.compare()."""

    @pytest.fixture
    def mock_arrhenius_nodes(self):
        """Create mock Arrhenius nodes for testing."""
        nodes = []
        for i in range(2):
            node = type("MockArrheniusNode", (), {})()
            node.name = f"arrhenius_node_{i}"
            node.activation_energy = {
                "[Li+]": {"mean": 0.3 + i * 0.05, "std": 0.02},
                "EC": {"mean": 0.5 + i * 0.03, "std": 0.03},
            }
            node.pre_exponential_factor = {
                "[Li+]": {"mean": 1e-3 * (1 + i * 0.1), "std": 1e-4},
                "EC": {"mean": 5e-4 * (1 + i * 0.08), "std": 5e-5},
            }
            nodes.append(node)
        return nodes

    def test_compare_basic(self, mock_arrhenius_nodes):
        """Test basic comparison of Arrhenius nodes."""
        from massband.diffusion.arrhenius import KinisiDiffusionArrhenius

        result = KinisiDiffusionArrhenius.compare(*mock_arrhenius_nodes)

        assert "figures" in result
        figures = result["figures"]

        assert "activation_energies" in figures
        assert "pre_exponential_factors" in figures

    def test_compare_with_labels(self, mock_arrhenius_nodes):
        """Test comparison with custom labels."""
        from massband.diffusion.arrhenius import KinisiDiffusionArrhenius

        labels = ["Exp 1", "Exp 2"]
        result = KinisiDiffusionArrhenius.compare(*mock_arrhenius_nodes, labels=labels)

        assert "figures" in result
        assert len(result["figures"]) == 2
