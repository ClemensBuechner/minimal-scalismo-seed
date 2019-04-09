package femurProject

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry.{Point, _3D}
import scalismo.mesh.TriangleMesh3D
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}

object IterativeClosestPoint {
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

  val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3),
    DenseMatrix.eye[Double](3))

  def nonrigidICP(movingMesh: TriangleMesh3D, staticMesh: TriangleMesh3D,
                  model: StatisticalMeshModel, ptIds: Seq[PointId], numberOfIterations: Int)
  : TriangleMesh3D = {

    if (numberOfIterations == 0) movingMesh
    else {
      val correspondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
      val transformed = ModelFitting.fit(correspondences, model, noise)

      nonrigidICP(transformed, staticMesh, model, ptIds, numberOfIterations - 1)
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
}
