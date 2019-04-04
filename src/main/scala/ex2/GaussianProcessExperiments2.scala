package ex2

import java.io.File

import scalismo.common.{Field, NearestNeighborInterpolator, RealSpace, VectorField}
import scalismo.geometry.{EuclideanVector, Point, SquareMatrix, _3D}
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object GaussianProcessExperiments2 {

  def main(args: Array[String]): Unit = {

    implicit val rng: Random = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val reference = MeshIO.readMesh(new File("datasets/femur.stl")).get
//    ui.show(reference, "reference")

    val group11 = ui.createGroup("experiment 1.1")
    ui.show(group11, meshModelFromKernel(reference, createKernel(10.0, 50.0)), "s=10/l=50")
    val group12 = ui.createGroup("experiment 1.2")
    ui.show(group12, meshModelFromKernel(reference, createKernel(100.0, 500.0)), "s=100/l=500")
    val group13 = ui.createGroup("experiment 1.3")
    ui.show(group13, meshModelFromKernel(reference, createKernel(10.0, 100.0)), "s=10/l=100")
    val group14 = ui.createGroup("experiment 1.4")
    ui.show(group14, meshModelFromKernel(reference, createKernel(100.0, 1000.0)), "s=100/l=1000")
//
//    val group21 = ui.createGroup("experiment 2.1")
//    ui.show(group21, meshModelFromKernel(reference, createKernel(10.0, 400.0)).truncate(1), "trunc 1")
//    val group22 = ui.createGroup("experiment 2.2")
//    ui.show(group22, meshModelFromKernel(reference, createKernel(10.0, 400.0)).truncate(10), "trunc 10")
//    val group23 = ui.createGroup("experiment 2.3")
//    ui.show(group23, meshModelFromKernel(reference, createKernel(10.0, 400.0)).truncate(100), "trunc 100")
//    val group24 = ui.createGroup("experiment 2.4")
//    ui.show(group24, meshModelFromKernel(reference, createKernel(10.0, 400.0)).truncate(200), "trunc 200")
//
//    val group3 = ui.createGroup("experiment 3")
//    ui.show(group3, meshModelFromKernel(reference, createKernelScaled(10.0, 400.0)), "scaled z")
//
//    val group4 = ui.createGroup("experiment 4")
//    ui.show(group4, meshModelFromKernel(reference, createKernel(10.0, 50.0) + createKernel(100.0, 500.0)), "combine")
  }

  def createKernel(s: Double, l: Double): DiagonalKernel[_3D] = {
    val gaussKernel: PDKernel[_3D] = GaussianKernel(l) * s
    DiagonalKernel(gaussKernel, gaussKernel, gaussKernel)
  }

  def createKernelScaled(s: Double, l: Double): DiagonalKernel[_3D] = {
    val gaussKernel: PDKernel[_3D] = GaussianKernel(l) * s
    val gaussKernel2: PDKernel[_3D] = GaussianKernel(l * 10) * 10 * s
    DiagonalKernel(gaussKernel, gaussKernel, gaussKernel2)
  }

  def meshModelFromKernel(referenceMesh: TriangleMesh[_3D], kernel: MatrixValuedPDKernel[_3D]):StatisticalMeshModel = {
    implicit val rng: Random = scalismo.utils.Random(42)
    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector(0, 0, 0))
    val gp = GaussianProcess(zeroMean, kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh.pointSet,
      gp,
      0.01,
      NearestNeighborInterpolator()
    )
    StatisticalMeshModel(referenceMesh, lowRankGP)
  }
}

/*

1)
To base our model on reality

2)
Smoother => less basis functions
Less complexity is easier to describe

3)
Lossy information, but easier to compute
+less memory, efficient
-less information

4)
Problematic if certain shapes to not occur within the limited samples. Lower rank process leads to less variance in our model.
- Handmade kernels
- More samples

*/