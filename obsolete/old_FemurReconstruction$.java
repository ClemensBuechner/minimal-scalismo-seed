package femurProject;

object old_FemurReconstruction {

  def main(args: Array[String]): Unit = {

    implicit val rng: Random = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val ref = MeshIO.readMesh(new File("datasets/femur.stl")).get
    val refLMs = LandmarkIO.readLandmarksJson[_3D](new File("datasets/femur.json")).get
    val refLMpts = landmarksToPoints(refLMs)

    val files = new File("data/femora/aligned/").listFiles()
    val dataset = files.map { f => MeshIO.readMesh(f).get }
    val lmsFiles = new File("data/femora/alignedLandmarks/").listFiles()
    val lms = lmsFiles.map { f => LandmarkIO.readLandmarksJson[_3D](f).get }

    val kernel = createKernel(10.0, 50.0) + createKernel(100.0, 500.0)
    val ssm = shapeModelFromKernel(ref, kernel)

    val sampler = UniformMeshSampler3D(ssm.referenceMesh, numberOfPoints = 5000)
    val points: Seq[Point[_3D]] = sampler.sample().map(pointWithProbability =>
      pointWithProbability._1)
    val pointIds = points.map(p => ssm.referenceMesh.pointSet.findClosestPoint(p).id)

    val defFields = dataset.indices.map { i: Int =>

      val warpedRef = warpMesh(ref, refLMpts, landmarksToPoints(lms(i)))
      val defField = computeDeformationField(dataset(i), warpedRef, ssm)
      println("generated deformation " + (i + 1) + " of " + dataset.length)
      defField
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val contiuousField = defFields.map(f => f.interpolate(interpolator))
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(ref.pointSet, contiuousField)
    val model = StatisticalMeshModel(ref, gp.interpolate(interpolator))

    ui.show(model, "mean")
  }

  def createKernel(s: Double, l: Double): DiagonalKernel[_3D] = {
    val gaussKernel: PDKernel[_3D] = GaussianKernel(l) * s
    DiagonalKernel(gaussKernel, gaussKernel, gaussKernel)
  }

  def shapeModelFromKernel(referenceMesh: TriangleMesh[_3D], kernel: MatrixValuedPDKernel[_3D])
  : StatisticalMeshModel = {
    implicit val rng: Random = scalismo.utils.Random(42)
    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector(0, 0, 0))
    val gp = GaussianProcess(zeroMean, kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh.pointSet,
      gp,
      0.01,
      NearestNeighborInterpolator()
    )
    StatisticalMeshModel(referenceMesh, lowRankGP)
  }

  def computeDeformationField(mesh1: TriangleMesh3D, mesh2: TriangleMesh3D,
                              ssm: StatisticalMeshModel): DiscreteField[_3D,
    UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = {

    val ids = (0 until mesh1.pointSet.numberOfPoints by 50).map(i => PointId(i))
    val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3))
    val aligned: TriangleMesh[_3D] = nonrigidICP(mesh1, mesh2, ssm, noise, ids, 150)
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

      p + weightedVecs.reduce { (a, b) => a + b }
    }.toIndexedSeq

    TriangleMesh3D(warpedPts, mesh.triangulation)
  }

  def attributeCorrespondences(movingMesh: TriangleMesh[_3D], staticMesh: TriangleMesh[_3      val weightedVecs = vectors.indices.map { i: Int => vectors(i) * weights(i) }D],
                               ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {

    ptIds.map { id: PointId =>
      val pt = movingMesh.pointSet.point(id)
      val closestPointOnMesh2 = staticMesh.pointSet.findClosestPoint(pt).point
      (id, closestPointOnMesh2)
    }
  }

  //  def ICPRigidAlign(movingMesh: TriangleMesh[_3D], staticMesh: TriangleMesh[_3D],
  //                    ptIds: Seq[PointId], numberOfIterations: Int): TriangleMesh[_3D] = {
  //
  //    if (numberOfIterations == 0) {
  //      movingMesh
  //    } else {
  //      val correspondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
  //      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences,
  //        center = Point(0, 0, 0))
  //      val transformed = movingMesh.transform(transform)
  //
  //      ICPRigidAlign(transformed, staticMesh, ptIds, numberOfIterations - 1)
  //    }
  //  }

  def nonrigidICP(movingMesh: TriangleMesh[_3D], staticMesh: TriangleMesh[_3D],
                  model: StatisticalMeshModel, noise: MultivariateNormalDistribution,
                  ptIds: Seq[PointId], numberOfIterations: Int): TriangleMesh[_3D] = {
    if (numberOfIterations == 0) movingMesh
    else {
      val correspondences = attributeCorrespondences(movingMesh, staticMesh, ptIds)
      val transformed = fitModel(correspondences, model, noise)

      nonrigidICP(transformed, staticMesh, model, noise, ptIds, numberOfIterations - 1)
    }
  }

  def fitModel(correspondences: Seq[(PointId, Point[_3D])], model: StatisticalMeshModel,
               noise: MultivariateNormalDistribution): TriangleMesh[_3D] = {
    val regressionData = correspondences.map(correspondence =>
      (correspondence._1, correspondence._2, noise)
    )
    val posterior = model.posterior(regressionData.toIndexedSeq)
    posterior.mean
  }
}
