package femurProject

import java.io.File

import breeze.interpolation.LinearInterpolator
import org.apache.commons.math3.analysis.interpolation.FieldHermiteInterpolator
import scalismo.common.{DiscreteField, NearestNeighborInterpolator, PointId, UnstructuredPointsDomain}
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, GaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object FemurReconstruction {

  def main(args: Array[String]): Unit = {
    implicit val rng: Random = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val files = new File("data/femora/aligned/").listFiles()
    val dataset = files.map { f => MeshIO.readMesh(f).get }
    val lmsFiles = new File("data/femora/landmarks/").listFiles()
    val lms: Array[Seq[Landmark[_3D]]] = files.map { f => LandmarkIO.readLandmarksJson[_3D](f).get }

    val reference = dataset.head
    val refLm: Seq[Landmark[_3D]] = lms.head
    val refLmPts: IndexedSeq[Point[_3D]] = refLm.map { lm => lm.point }.toIndexedSeq

    val defFields = (1 until dataset.length).map { i: Int =>
      val gp = getGPfromLandmarks(refLmPts, refLm, lms(i))
      val mesh: TriangleMesh[_3D] = dataset(i).pointSet.pointIds.map { id: PointId =>
        gp.marginal(reference.pointSet)
      }

      val ids = (0 until m.pointSet.numberOfPoints by 50).map(i => PointId(i))
      val aligned: TriangleMesh[_3D] = ICPRigidAlign(m, reference, ids, 150)
      val deformationVectors = aligned.pointSet.pointIds.map { id: PointId =>
        val p = m.pointSet.point(id)
        p - reference.pointSet.findClosestPoint(p).point
      }.toIndexedSeq

      println("finished alignment")
      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](m.pointSet,
        deformationVectors)
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val contiuousField = defFields.map(f => f.interpolate(interpolator))
    val gp1 = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, contiuousField)
    val model1 = StatisticalMeshModel(reference, gp1.interpolate(interpolator))
    ui.show(model1, "mean")
  }

  def getGPfromLandmarks(reference: IndexedSeq[Point[_3D]], refLms: Seq[Landmark[_3D]],
                         lms: Seq[Landmark[_3D]]):
  DiscreteLowRankGaussianProcess[_3D, IndexedSeq[Point[_3D]], EuclideanVector[_3D]] = {
    val defField = lms.indices.map { i: Int => refLms(i).point - lms(i).point }
    val discField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] =
      DiscreteField(UnstructuredPointsDomain(reference), defField)
    DiscreteLowRankGaussianProcess.createUsingPCA(reference,
      Seq(discField.interpolate(NearestNeighborInterpolator()))) // TODO: replace nearestNeighbourInterpolator
  }

  def attributeCorrespondences(movingMesh: TriangleMesh[_3D], staticMesh: TriangleMesh[_3D],
                               ptIds: Seq[PointId]): Seq[(Point[_3D], Point[_3D])] = {
    ptIds.map { id: PointId =>
      val pt = movingMesh.pointSet.point(id)
      val closestPointOnMesh2 = staticMesh.pointSet.findClosestPoint(pt).point
      (pt, closestPointOnMesh2)
    }
  }

  def ICPRigidAlign(movingMesh: TriangleMesh[_3D], staticMesh: TriangleMesh[_3D],
                    ptIds: Seq[PointId], numberOfIterations: Int): TriangleMesh[_3D] = {
    if (numberOfIterations == 0) movingMesh
    else {
      val correspondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences,
        center = Point(0, 0, 0))
      val transformed = movingMesh.transform(transform)

      ICPRigidAlign(transformed, staticMesh, ptIds, numberOfIterations - 1)
    }
  }
}
