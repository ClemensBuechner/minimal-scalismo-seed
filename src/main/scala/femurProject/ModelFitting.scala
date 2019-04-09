package femurProject

import scalismo.common.PointId
import scalismo.geometry.{Point, _3D}
import scalismo.mesh.TriangleMesh3D
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}

object ModelFitting {

  def fit(correspondences: Seq[(PointId, Point[_3D])], model: StatisticalMeshModel,
               noise: MultivariateNormalDistribution): TriangleMesh3D = {

    val regressionData = correspondences.map(correspondence =>
      (correspondence._1, correspondence._2, noise)
    )
    val posterior = model.posterior(regressionData.toIndexedSeq)
    posterior.mean
  }
}
