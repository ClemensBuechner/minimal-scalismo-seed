package ex2

import java.io.File

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, GaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object GaussianProcessExperiments {

  def main(args: Array[String]): Unit = {

    implicit val rng: Random = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val files = new File("data/femora/aligned/").listFiles()
    val dataset = files.map{f => MeshIO.readMesh(f).get}

    val reference = dataset.head
    val defFields = dataset.tail.map { m =>
      val deformationVectors = m.pointSet.pointIds.map { id : PointId =>
        val p = m.pointSet.point(id)
        p - reference.pointSet.findClosestPoint(p).point
      }.toIndexedSeq

      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](m.pointSet, deformationVectors)
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val contiuousField = defFields.map( f => f.interpolate(interpolator))
    val gp1 = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, contiuousField)
    val model1 = StatisticalMeshModel(reference, gp1.interpolate(interpolator))
    ui.show(model1, "mean")

    val l = 40.0
    val s = 10.0
    val zeroMean: VectorField[_3D, _3D] = VectorField(RealSpace[_3D], (pt: Point[_3D]) => EuclideanVector(0, 0, 0))
    val matrixValuedGaussian: MatrixValuedPDKernel[_3D] = DiagonalKernel(GaussianKernel[_3D](l) * s, 3)
    val gp2 = GaussianProcess(zeroMean, matrixValuedGaussian)
//    val model2 = StatisticalMeshModel(reference, )
//    ui.show(gp2.sampleAtPoints(reference), "sample")


  }

  def createKernel(s: Double, l: Double): (Point[_3D], Point[_3D]) => Double = {
    (p1: Point[_3D], p2: Point[_3D]) => {
      val norm = (p1 - p2).norm2
      s * scala.math.exp(- norm / (l * l))
    }
  }
}
