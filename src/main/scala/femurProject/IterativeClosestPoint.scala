package femurProject

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry.{Point, _3D}
import scalismo.mesh.TriangleMesh3D
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}

object IterativeClosestPoint {

  val maxIterations = 30

  def ICPRigidAlign(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D, ptIds: Seq[PointId],
                    numberOfIterations: Int): TriangleMesh3D = {

    if (numberOfIterations == 0) {
      movingMesh
    } else {
      val idCorrespondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
      val ptCorrespondences = idCorrespondences.map { c => (movingMesh.pointSet.point(c._1), c._2) }
      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(ptCorrespondences, center =
        Point(0, 0, 0))
      val transformed = movingMesh.transform(transform)

      ICPRigidAlign(transformed, staticMesh, ptIds, numberOfIterations - 1)
    }
  }

  def nonrigidICP(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                  model: StatisticalMeshModel, ptIds: Seq[PointId]): TriangleMesh3D = {

    nonrigidICP(movingMesh, staticMesh, model, ptIds, 1000, 0)
  }

  def nonrigidICP(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                  model: StatisticalMeshModel, ptIds: Seq[PointId], error: Double,
                  recursionDepth: Int): TriangleMesh3D = {

    if (recursionDepth > maxIterations) {
      println("ICP reacht maximal recurison depth.")
      movingMesh
    } else {
      val correspondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
      val transformed = fit(correspondences, model)

      val newError = scalismo.mesh.MeshMetrics.hausdorffDistance(transformed, staticMesh)
      if (Math.abs(error - newError) < 1e-5) {
        println("ICP Converged after " + recursionDepth + " iterations.")
        movingMesh
      } else nonrigidICP(transformed, staticMesh, model, ptIds, newError, recursionDepth + 1)
    }
  }

  def partialICP(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                 model: StatisticalMeshModel, ptIds: Seq[PointId]): TriangleMesh3D = {

    partialICP(movingMesh, staticMesh, model, ptIds, 1000, 0)
  }

  def partialICP(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                 model: StatisticalMeshModel, ptIds: Seq[PointId], error: Double,
                 recursionDepth: Int): TriangleMesh3D = {

    if (recursionDepth > maxIterations) {
      println("ICP reached maximal recursion depth.")
      movingMesh
    } else {
      val correspondences = partialCorrespondences(movingMesh, staticMesh, ptIds)
      val transformed = fit(correspondences, model)

      val newError = scalismo.mesh.MeshMetrics.hausdorffDistance(transformed, staticMesh)
      if (Math.abs(error - newError) < 1e-5) {
        println("ICP Converged after " + recursionDepth + " iterations.")
        movingMesh
      } else partialICP(transformed, staticMesh, model, ptIds, newError, recursionDepth + 1)
    }
  }

  def attributeCorrespondences(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                               ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {

    ptIds.map { id: PointId =>
      val pt = movingMesh.pointSet.point(id)
      val closestPoint = staticMesh.pointSet.findClosestPoint(pt).point
      (id, closestPoint)
    }
  }

  def partialCorrespondences(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                             ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {

    ptIds.map { id: PointId =>
      val pt = staticMesh.pointSet.point(id)
      val closestPointId = movingMesh.pointSet.findClosestPoint(pt).id
      (closestPointId, pt)
    }
  }

  val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3),
    DenseMatrix.eye[Double](3))

  def fit(correspondences: Seq[(PointId, Point[_3D])], model: StatisticalMeshModel)
  : TriangleMesh3D = {

    val regressionData = correspondences.map(correspondence =>
      (correspondence._1, correspondence._2, noise)
    )
    val posterior = model.posterior(regressionData.toIndexedSeq)
    posterior.mean
  }
}
