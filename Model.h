#ifndef MODEL
#define MODEL

#include <boost/mpi.hpp>
#include "repast_hpc/Schedule.h"
#include "repast_hpc/Properties.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/SharedDiscreteSpace.h"
#include "repast_hpc/GridComponents.h"
#include "repast_hpc/Random.h"
#include <math.h>
#include "Household.h"

#define NUMBER_OF_YEARS 551

class AnasaziModel {
private:
	int year;
	int stopAt;
	int boardSizeX, boardSizeY, procX, procY, bufferSize;
	int randomSeed;
	int houseID = 0;

	std::ofstream out;
	struct Parameters
	{
		int startYear;
		int endYear;
		int maxStorageYear;
		int maxStorage;
		int householdNeed;
		int minFissionAge;
		int maxFissionAge;
		int minDeathAge;
		int maxDeathAge;
		int maxDistance;
		int initMinCorn;
		int initMaxCorn;
		double annualVariance;
		double spatialVariance;
		double fertilityProbability;
		double harvestAdjustment;
		double maizeStorageRatio;
	} param;

	struct PDSI
	{
		int year;
		double pdsiGeneral;
		double pdsiNorth;
		double pdsiMid;
		double pdsiNatural;
		double pdsiUpland;
		double pdsiKinbiko;
	} pdsi[NUMBER_OF_YEARS];

	struct Hydro
	{
		int year;
		double hydroGeneral;
		double hydroNorth;
		double hydroMid;
		double hydroNatural;
		double hydroUpland;
		double hydroKinbiko;
	} hydro[NUMBER_OF_YEARS];
	//======component4 added new struct
	//======add new struct
	struct InfluenceParams {
		int    windowL;       // L：记忆年数
		double rho;           // 影响衰减

		double capPerYear;    // 每年最多跟随比例（cap）

		// ------ 新增：Feature 3 logistic 权重和阈值 ------
		double k_d;
		double k_b;
		double k_a;
		double k_c;
		double k_e;
		double probBase;      // probability_base
	};
	InfluenceParams infl;

	// Feature 2: Settlement attachment 参数
	struct SettlementParams {
		double lambda;   // settlement.lambda
		double alpha;    // A_total_i 中 α
		double a;        // Laplace 平滑 a
		double b;        // Laplace 平滑 b
	};
	SettlementParams settle;

	// Feature 1: 社会风险共担 / 声誉参数（未来借贷模块用）
	struct SocialParams {
		double ks;       // kinship 权重 k_s
		double k_up;     // 声誉增加
		double k_down;   // 声誉惩罚
	};
	SocialParams social;


	// 记录每个迁出者的 ID + 离开时所在坐标
	struct MigrantRecord {
		repast::AgentId id;
		std::vector<int> location;   // {x, y}
	};
	//======add new vector for migrantsHistory
	std::vector<std::vector<MigrantRecord>> migrantsHistory;
	//15 仅记录自然迁出者（natural migrants）的 ID + 离开时坐标（过去 windowL 年）
	std::vector<std::vector<MigrantRecord>> naturalMigrantsHistory;



	const int yieldLevels[5][4] = { {617, 514, 411, 642},
		{719, 599, 479, 749},
		{821, 684, 547, 855},
		{988, 824, 659, 1030},
		{1153, 961, 769, 1201} };

	bool existStreams;
	bool existAlluvium;
	repast::Properties* props;
	repast::SharedContext<Household> context;
	repast::SharedContext<Location> locationContext;	//Need to confirm this line
	repast::SharedDiscreteSpace<Household, repast::StrictBorders, repast::SimpleAdder<Household> >* householdSpace;
	repast::SharedDiscreteSpace<Location, repast::StrictBorders, repast::SimpleAdder<Location> >* locationSpace;
	repast::DoubleUniformGenerator* fissionGen;// = repast::Random::instance()->createUniDoubleGenerator(0,1);
	repast::IntUniformGenerator* deathAgeGen;// = repast::Random::instance()->createNormalGenerator(25,5);
	repast::NormalGenerator* yieldGen;// = repast::Random::instance()->createNormalGenerator(0,sqrt(0.1));
	repast::NormalGenerator* soilGen;// = repast::Random::instance()->createNormalGenerator(0,sqrt(0.1));
	repast::IntUniformGenerator* initAgeGen;// = repast::Random::instance()->createUniIntGenerator(0,29);
	repast::IntUniformGenerator* initMaizeGen;// = repast::Random::instance()->createUniIntGenerator(1000,1600);

	//component4 added 
	// 计算两个户之间的欧氏距离（基于 house location）
	double distanceBetween(const Household* a, const Household* b) const;

	// 从 ID 获取当前仍在 context 中的 Household 指针（如果还存在的话）
	Household* getHouseholdById(const repast::AgentId& id);

	// 下面这几个以后可以用真实的 D_i, B_i, A_total, Cstability 替代
	double getDeficitRatio(const Household* h) const;     // D_i
	double getBufferingRatio(const Household* h) const;   // B_i
	double getAttachment(const Household* h) const;       // A_total_i
	double getStability(const Household* h) const;        // Cstability_i
	// ---------- Stage 2 storage ----------
	std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, double>> debt_ij; // borrower -> (lender -> amount)

	// child -> parent（用于 K_ij；K_ij=1 当且仅当 i 是 j 的父子之一）
	std::unordered_map<std::uint64_t, std::uint64_t> parentOf;

	// ---------- Stage 2 helpers ----------
	std::uint64_t keyOf(const repast::AgentId& id) const;
	bool isKin(std::uint64_t a, std::uint64_t b) const;
	double clamp01(double x) const;
	double tieStrength(Household* borrower, Household* lender) const;   // S_ij
	double randn01() const;                                             // N(0,1)
	double truncNormal(double mu, double sigma, double lo, double hi, int maxTry = 20) const;

	// Stage 2 main
	void processSocialRiskPooling(std::vector<Household*>& highDebtHouseholds);

	void logCheckSummary(int naturalCount, int followerCount, int highDebtCount, int survivorsCount);
	void logCheckSamples(const std::vector<Household*>& survivors);

public:
	AnasaziModel(std::string propsFile, int argc, char** argv, boost::mpi::communicator* comm);
	~AnasaziModel();
	void initAgents();
	void initSchedule(repast::ScheduleRunner& runner);
	void doPerTick();
	void readCsvMap();
	void readCsvWater();
	void readCsvPdsi();
	void readCsvHydro();
	int yieldFromPdsi(int zone, int maizeZone);
	double hydroLevel(int zone);
	void checkWaterConditions();
	void writeOutputToFile();
	void updateLocationProperties();
	void updateHouseholdProperties();
	bool fieldSearch(Household* household);
	void removeHousehold(Household* household);
	bool relocateHousehold(Household* household);
	//add new functions
	// Feature 3: Follow-migration
	double computeExposure(Household* i);   // 使用 migrantsHistory 和空间距离
	bool   followMigrationDecision(const Household* h, double E);

	void processSocialInfluenceMigration(
		const std::vector<repast::AgentId>& natural_migrants_t,
		std::vector<repast::AgentId>& followers_t);


	void removeHouseholdById(const repast::AgentId& id);



};

#endif
