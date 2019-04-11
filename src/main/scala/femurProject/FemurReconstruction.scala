package femurProject

import java.awt.Color
import java.io.File

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh3D
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{
  DiscreteLowRankGaussianProcess, GaussianProcess,
  LowRankGaussianProcess, StatisticalMeshModel
}
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

    val sampler = UniformMeshSampler3D(model.referenceMesh, numberOfPoints = 5000)
    val points = sampler.sample().map { pointWithProbability => pointWithProbability._1 }
    val pointIds = points.map { pt => model.referenceMesh.pointSet.findClosestPoint(pt).id }
    println("Finished sampling points on the mesh.")

    val referenceLMpts = landmarksToPoints(referenceLandmarks)
    val defFields = targets.indices.map { i: Int =>
      //    val defFields = (0 until 10).map { i: Int =>
      val target = targets(i)
      val targetLMpts = landmarksToPoints(targetLandmarks(i))
      val warp = warpMesh(reference, referenceLMpts, targetLMpts)
      val aligned = IterativeClosestPoint.nonrigidICP(warp, target, model, pointIds, 150)
      // TODO: play with the number of iterations

      //      val targetView = ui.show(target, "target")
      //      val alignedView = ui.show(aligned, "aligned")
      //      alignedView.color = Color.RED

      val ids = reference.pointSet.pointIds.map { id => (id, id) }.toIndexedSeq
      val defField = computeDeformationField(reference, aligned, ids)

      //      StdIn.readLine()
      //      targetView.remove()
      //      alignedView.remove()

      println("Generated " + (i + 1) + " of " + targets.length + " deformation fields.")
      defField
    }
    StdIn.readLine("Finished calculating model. Go on?")

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousField = defFields.map { f => f.interpolate(interpolator) }
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousField)
    val finalModel = StatisticalMeshModel(reference, gp.interpolate(interpolator))

    ui.show(finalModel, "mean")
    StdIn.readLine("Finished computing the reconstruction model.")

    val partialFiles = new File("data/femora/partial/").listFiles()
    val partials = partialFiles.map { f => MeshIO.readMesh(f).get }
    println("Loaded dataset of partial bones.")

    val referenceView = ui.show(reference, "reference")
    partials.indices.map { i: Int =>
      StdIn.readLine("Reconstruct first partial: press [enter].")
      val sampler = UniformMeshSampler3D(partials(i), numberOfPoints = 3000)
      val points = sampler.sample().map { pointWithProbability => pointWithProbability._1 }
      val pointIds = points.map { pt => partials(i).pointSet.findClosestPoint(pt).id }
      val aligned = IterativeClosestPoint.nonrigidICP(partials(i), reference, model, pointIds, 150)
      val ids = pointIds.map { id =>
        (reference.pointSet.findClosestPoint(aligned.pointSet.point(id)).id, id)
      }.toIndexedSeq
      val defField = computeDeformationField(reference, partials(i), ids)

      val partialView = ui.show(aligned, "partial_aligned")
      StdIn.readLine()
      partialView.remove()

      aligned
    }
    referenceView.remove()
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
      0.05, NearestNeighborInterpolator()) // TODO: change tolerance to smaller value
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

  def computeDeformationField(from: TriangleMesh3D, to: TriangleMesh3D, ids: IndexedSeq[(PointId,
    PointId)]): DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = {

    val deformationVectors = ids.map { id =>
      to.pointSet.point(id._2) - from.pointSet.point(id._1)
    }.toIndexedSeq

    DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](from.pointSet,
      deformationVectors)
  }
}
