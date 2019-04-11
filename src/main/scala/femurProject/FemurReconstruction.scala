package femurProject

import java.awt.Color
import java.io.File

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh3D
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import scala.io.StdIn
;

object FemurReconstruction {

  scalismo.initialize()
  val ui = ScalismoUI()

  def main(args: Array[String]): Unit = {

    implicit val rng: Random = scalismo.utils.Random(42)

    val reference = MeshIO.readMesh(new File("datasets/femur.stl")).get
    val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new File("datasets/femur.json")).get
    println("Loaded reference.")

    val files = new File("data/femora/aligned/").listFiles()
    val targets = files.map { f => MeshIO.readMesh(f).get }
    val landmarkFiles = new File("data/femora/alignedLandmarks/").listFiles()
    val targetLandmarks = landmarkFiles.map { f => LandmarkIO.readLandmarksJson[_3D](f).get }
    println("Loaded dataset of targets.")

    val kernel = createKernel(10.0, 50.0) + createKernel(100.0, 500.0)
    val model = shapeModelFromKernel(reference, kernel)
    println("Generated shape model from kernel.")

    val sampler = UniformMeshSampler3D(model.referenceMesh, numberOfPoints = 8000)
    val points = sampler.sample().map { pointWithProbability => pointWithProbability._1 }
    val pointIds = points.map { pt => model.referenceMesh.pointSet.findClosestPoint(pt).id }
    println("Finished sampling points on the mesh.")

    val referenceLMpts: IndexedSeq[Point[_3D]] = landmarksToPoints(referenceLandmarks)
//    val defFields = targets.indices.map { i: Int =>
    val defFields = (0 until 5).map { i: Int =>
      val targetLMpts = landmarksToPoints(targetLandmarks(i))
      val defField = computeDeformationField(reference, referenceLMpts, targets(i), targetLMpts,
        model, pointIds)
      println("Generated " + (i + 1) + " of " + targets.length + " deformation fields.")
      defField
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousField = defFields.map(f => f.interpolate(interpolator))
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousField)
    val finalModel = StatisticalMeshModel(reference, gp.interpolate(interpolator))

    ui.show(finalModel, "mean")
  }

  def landmarksToPoints(lms: Seq[Landmark[_3D]]): IndexedSeq[Point[_3D]] = {

    lms.map { lm => lm.point }.toIndexedSeq
  }

  def createKernel(s: Double, l: Double): DiagonalKernel[_3D] = {

    val gaussKernel: PDKernel[_3D] = GaussianKernel(l) * s
    DiagonalKernel(gaussKernel, gaussKernel, gaussKernel)
  }

  def shapeModelFromKernel(referenceMesh: TriangleMesh3D, kernel: MatrixValuedPDKernel[_3D])
  : StatisticalMeshModel = {

    implicit val rng: Random = scalismo.utils.Random(42)
    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector(0, 0, 0))
    val gp = GaussianProcess(zeroMean, kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(referenceMesh.pointSet, gp,
      0.01, NearestNeighborInterpolator()) // TODO: change tolerance to smaller value
    StatisticalMeshModel(referenceMesh, lowRankGP)
  }

  def warpMesh(mesh: TriangleMesh3D, orig: IndexedSeq[Point[_3D]],
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

  def computeDeformationField(moving: TriangleMesh3D, movingLandmarks: IndexedSeq[Point[_3D]],
                              target: TriangleMesh3D, targetLandmarks: IndexedSeq[Point[_3D]],
                              model: StatisticalMeshModel, ptIds: IndexedSeq[PointId])
  : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = {

    val movingView = ui.show(moving, "moving")
    movingView.color = Color.GREEN
    val targetView = ui.show(target, "target")
    val warpedMovingMesh = warpMesh(moving, movingLandmarks, targetLandmarks)
    val aligned = IterativeClosestPoint.nonrigidICP(warpedMovingMesh, target, model, ptIds,
      150) //
    // TODO: play with the number of iterations
    val alignedView = ui.show(aligned, "aligned")
    alignedView.color = Color.RED
    val deformationVectors = moving.pointSet.pointIds.map { id: PointId =>
      aligned.pointSet.point(id) - moving.pointSet.point(id)
    }.toIndexedSeq

    StdIn.readLine()
    targetView.remove()
    movingView.remove()
    alignedView.remove()

    DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](moving.pointSet,
      deformationVectors)
  }
}
