package femurProject

import java.io.File

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object FemurReconstruction {

  def main(args: Array[String]): Unit = {

    implicit val rng: Random = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val model = computeModel()
    ui.show(model, "mean")
  }

  def computeModel(): StatisticalMeshModel = {

    val ref = MeshIO.readMesh(new File("datasets/femur.stl")).get
    val refLMs = LandmarkIO.readLandmarksJson[_3D](new File("datasets/femur.json")).get
    val refLMpts = landmarksToPoints(refLMs)

    val files = new File("data/femora/aligned/").listFiles()
    val dataset = files.map { f => MeshIO.readMesh(f).get }
    val lmsFiles = new File("data/femora/alignedLandmarks/").listFiles()
    val lms = lmsFiles.map { f => LandmarkIO.readLandmarksJson[_3D](f).get }

    val defFields = dataset.indices.map { i: Int =>
      val warpedRef = warpMesh(ref, refLMpts, landmarksToPoints(lms(i)))
      val defField = computeDeformationField(dataset(i), warpedRef)
      println("generated deformation " + (i + 1) + " of " + dataset.length)
      defField
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val contiuousField = defFields.map(f => f.interpolate(interpolator))
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(ref.pointSet, contiuousField)
    StatisticalMeshModel(ref, gp.interpolate(interpolator))
  }

  def computeDeformationField(mesh1: TriangleMesh3D, mesh2: TriangleMesh3D): DiscreteField[_3D,
    UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = {

    val ids = (0 until mesh1.pointSet.numberOfPoints by 50).map(i => PointId(i))
    val aligned: TriangleMesh[_3D] = ICPRigidAlign(mesh1, mesh2, ids, 150)
    val deformationVectors = aligned.pointSet.points.map { p: Point[_3D] =>
      p - mesh2.pointSet.findClosestPoint(p).point
    }.toIndexedSeq

    DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](mesh1.pointSet,
      deformationVectors)
  }

  def landmarksToPoints(lms: Seq[Landmark[_3D]]): IndexedSeq[Point[_3D]] = {
    lms.map { lm => lm.point }.toIndexedSeq
  }

  def warpMesh(mesh: TriangleMesh[_3D], orig: IndexedSeq[Point[_3D]],
               target: IndexedSeq[Point[_3D]]): TriangleMesh3D = {

    val vectors = orig.indices.map { i: Int => target(i) - orig(i) }
    val warpedPts = mesh.pointSet.points.map { p =>
      val dists = orig.map { o => (p - o).norm2 }
      val distSum = dists.sum
      val weights = dists.map { d => d / distSum }
      val weightedVecs = vectors.indices.map { i: Int => vectors(i) * weights(i) }
      p + weightedVecs.reduce { (a, b) => a + b }
    }.toIndexedSeq

    TriangleMesh3D(warpedPts, mesh.triangulation)
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

    if (numberOfIterations == 0) {
      movingMesh
    } else {
      val correspondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences,
        center = Point(0, 0, 0))
      val transformed = movingMesh.transform(transform)

      ICPRigidAlign(transformed, staticMesh, ptIds, numberOfIterations - 1)
    }
  }
}
