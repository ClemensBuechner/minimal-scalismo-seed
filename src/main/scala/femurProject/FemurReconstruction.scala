package femurProject

import java.awt.Color
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

  scalismo.initialize()
  val ui = ScalismoUI()

  def main(args: Array[String]): Unit = {

    implicit val rng: Random = scalismo.utils.Random(42)

    val model = computeModel()
    ui.show(model, "mean")
  }

  def computeModel(): StatisticalMeshModel = {

    // load data from which the model is generated
    val ref = MeshIO.readMesh(new File("datasets/femur.stl")).get
    val refLMs = LandmarkIO.readLandmarksJson[_3D](new File("datasets/femur.json")).get
    val refLMpts: IndexedSeq[Point[_3D]] = landmarksToPoints(refLMs)

    val files = new File("data/femora/aligned/").listFiles()
    val dataset = files.map { f => MeshIO.readMesh(f).get }
    val lmsFiles = new File("data/femora/alignedLandmarks/").listFiles()
    val lms = lmsFiles.map { f => LandmarkIO.readLandmarksJson[_3D](f).get }

    // compute deformation fields and interpolate continuous vector field from it
    val defFields = /*(0 until 5)*/ dataset.indices.map { i: Int =>
      //      val data = ui.show(dataset(i), "femora" + i)
      //      val warp = ui.show(warpedRef, "warp" + i)
      //      data.color = Color.ORANGE
      //      warp.color = Color.GREEN

      val ptIds = (0 until dataset(i).pointSet.numberOfPoints by 50).map { i: Int => PointId(i) } // TODO: play with value
      val defField = computeDeformationField((ref, refLMpts), (dataset(i),
        landmarksToPoints(lms(i))), ptIds)
      println("generated deformation " + (i + 1) + " of " + dataset.length)
      defField
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousField = defFields.map(f => f.interpolate(interpolator))
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(ref.pointSet, continuousField)
    StatisticalMeshModel(ref, gp.interpolate(interpolator))
  }

  def computeDeformationField(instance: (TriangleMesh3D, IndexedSeq[Point[_3D]]), target:
  (TriangleMesh3D, IndexedSeq[Point[_3D]]), ptIds: Seq[PointId]): DiscreteField[_3D,
    UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = {

    val t = ui.show(target._1, "target")
    val o = ui.show(instance._1, "original")
    t.color = Color.GREEN
    o.color = Color.RED

    val warp = warpMesh(instance._1, instance._2, target._2)
    val ids = (0 until instance._1.pointSet.numberOfPoints by 50).map(i => PointId(i))
    val aligned: TriangleMesh[_3D] = ICPRigidAlign(warp, target._1, ids, 80)
    val icp = ui.show(aligned, "icp instance")

    val deformationVectors = aligned.pointSet.pointIds.map { id: PointId =>
      val orig = instance._1.pointSet.point(id)
      val alig = aligned.pointSet.point(id)
      target._1.pointSet.findClosestPoint(alig).point - orig
    }.toIndexedSeq

    Thread.sleep(1000)
    icp.remove()
    o.remove()
    t.remove()
    DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](instance._1.pointSet,
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

//      if (numberOfIterations % 10 == 0) {
//        val instance = ui.show(transformed, "icp inistance")
//        Thread.sleep(2000)
//        instance.remove()
//      }

      ICPRigidAlign(transformed, staticMesh, ptIds, numberOfIterations - 1)
    }
  }
}
