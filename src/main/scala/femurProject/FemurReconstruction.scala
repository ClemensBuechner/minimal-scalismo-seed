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
import scalismo.ui.model
import scalismo.utils.Random

import scala.io.StdIn

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

    val kernel = createKernel(25.0, 25.0) + createKernelScaled(100.0, 500.0)
    val model = shapeModelFromKernel(reference, kernel)
    val kernelGroup = ui.createGroup("kernel model")
    val kernelModel = ui.show(kernelGroup, model, "kernel")
    println("Generated shape model from kernel.")

    val sampler = UniformMeshSampler3D(model.referenceMesh, numberOfPoints = 8000)
    val points = sampler.sample().map { pointWithProbability => pointWithProbability._1 }
    val pointIds = points.map { pt => model.referenceMesh.pointSet.findClosestPoint(pt).id }
    println("Finished sampling points on the mesh.")

    val registrationGroup = ui.createGroup("registration")
    //    val defFields = targets.indices.map { i: Int =>
    val defFields = (0 until 3).map { i: Int =>
      val target = targets(i)
      val registration = getRegistration("data/femora/deformations/" + i + ".ply", model,
        reference, referenceLandmarks, target, targetLandmarks(i), pointIds)

      val targetView = ui.show(registrationGroup, target, "target")
      val registrationView = ui.show(registrationGroup, registration, "aligned")
      registrationView.color = Color.RED

      val dist = scalismo.mesh.MeshMetrics.avgDistance(registration, target)
      val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(registration, target)
      println("Average Distance: " + dist)
      println("Hausdorff Distance: " + hausDist)

      val ids = reference.pointSet.pointIds.map { id => (id, id) }.toIndexedSeq
      val defField = computeDeformationField(reference, registration, ids)

      StdIn.readLine("Show next registration.")
      targetView.remove()
      registrationView.remove()

      println("Generated " + (i + 1) + " of " + targets.length + " registration fields.")
      defField
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousField = defFields.map { f => f.interpolate(interpolator) }
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousField)
    val finalModel = StatisticalMeshModel(reference, gp.interpolate(interpolator))

    val modelGroup = ui.createGroup("gp from deformations")
    ui.show(modelGroup, finalModel, "mean")

    val partialFiles = new File("data/femora/partial/").listFiles()
    val partials = partialFiles.map { f => MeshIO.readMesh(f).get }
    println("Loaded dataset of partial bones.")

    StdIn.readLine("Reconstruct first partial: press [enter].")
    val partialGroup = ui.createGroup("partials")
    partials.indices.map { i: Int =>

      val sampler = UniformMeshSampler3D(partials(i), numberOfPoints = 3000)
      val points = sampler.sample().map { pointWithProbability => pointWithProbability._1 }
      val pointIds = points.map { pt => partials(i).pointSet.findClosestPoint(pt).id }

      val partialView = ui.show(partialGroup, partials(i), "partial")
      partialView.color = Color.BLUE
      val aligned = IterativeClosestPoint.partialICP(finalModel.mean, partials(i), finalModel,
        pointIds, 30)

      val ids = pointIds.map { id =>
        (finalModel.mean.pointSet.findClosestPoint(aligned.pointSet.point(id)).id, id)
      }.toIndexedSeq
      //      val defField = computeDeformationField(reference, partials(i), ids)

      val partialAlignedView = ui.show(partialGroup, aligned, "partial_aligned")
      StdIn.readLine("Show next reconstruction.")
      partialView.remove()
      partialAlignedView.remove()

      aligned
    }
  }

  def getRegistration(filename: String, model: StatisticalMeshModel, moving: TriangleMesh3D,
                      movingLandmarks: Seq[Landmark[_3D]], target: TriangleMesh3D,
                      targetLandmarks: Seq[Landmark[_3D]], ids: Seq[PointId]): TriangleMesh3D = {

    val file: File = new File(filename)
    val registration: TriangleMesh3D = {
      if (file.exists()) {
        println("Loading mesh from " + filename)
        MeshIO.readMesh(file).get
      } else {
        println("Computing registration.")
        val lmCorrespondences = movingLandmarks.indices.map { i: Int =>
          val id = moving.pointSet.findClosestPoint(movingLandmarks(i).point).id
          (id, targetLandmarks(i).point)
        }
        val posterior = model.posterior(lmCorrespondences, 1)
        val registration = IterativeClosestPoint.nonrigidICP(moving, target, posterior, ids, 20)
        // TODO: play with the number of iterations
        MeshIO.writeMesh(registration, file)
        registration
      }
    }
    registration
  }

  def landmarksToPoints(lms: Seq[Landmark[_3D]]): IndexedSeq[Point[_3D]] = {

    lms.map {
      lm => lm.point
    }.toIndexedSeq
  }

  def createKernel(s: Double, l: Double): DiagonalKernel[_3D] = {

    val gaussKernel: PDKernel[_3D] = GaussianKernel(l) * s
    DiagonalKernel(gaussKernel, gaussKernel, gaussKernel)
  }

  def createKernelScaled(s: Double, l: Double): DiagonalKernel[_3D] = {
    val gaussKernel: PDKernel[_3D] = GaussianKernel(l) * s
    val gaussKernel2: PDKernel[_3D] = GaussianKernel(l * 2) * 2 * s
    DiagonalKernel(gaussKernel, gaussKernel, gaussKernel2)
  }

  def shapeModelFromKernel(referenceMesh: TriangleMesh3D, kernel: MatrixValuedPDKernel[_3D])
  : StatisticalMeshModel = {

    implicit val rng: Random = scalismo.utils.Random(42)
    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector(0, 0, 0))
    val gp = GaussianProcess(zeroMean, kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(referenceMesh.pointSet, gp,
      0.1, NearestNeighborInterpolator()) // TODO: change tolerance to smaller value
    StatisticalMeshModel(referenceMesh, lowRankGP)
  }

  def warpMesh(mesh: TriangleMesh3D, orig: IndexedSeq[Point[_3D]],
               target: IndexedSeq[Point[_3D]]): TriangleMesh3D = {

    val vectors = orig.indices.map {
      i: Int => target(i) - orig(i)
    }
    val warpedPts = mesh.pointSet.points.map {
      p =>
        val dists = orig.map {
          o => (p - o).norm2
        }
        val distSum = dists.sum
        val weights = dists.map {
          d => d / distSum
        }
        val weightedVecs = vectors.indices.map {
          i: Int => vectors(i) * weights(i)
        }
        p + weightedVecs.reduce {
          (a, b) => a + b
        }
    }.toIndexedSeq

    TriangleMesh3D(warpedPts, mesh.triangulation)
  }


  def computeDeformationField(from: TriangleMesh3D, to: TriangleMesh3D, ids: IndexedSeq[(PointId,
    PointId)]): DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = {

    val deformationVectors = ids.map {
      id =>
        to.pointSet.point(id._2) - from.pointSet.point(id._1)
    }.toIndexedSeq

    DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](from.pointSet,
      deformationVectors)
  }
}
