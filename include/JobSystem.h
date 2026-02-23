#pragma once
#include <vector>
#include <thread>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <mutex>

class JobSystem {
private:
	uint32_t numThreads;
	std::vector<std::thread> workers;
	std::atomic<bool> terminatePool{ false };

	std::function<void(uint32_t)> taskFunc;
	uint32_t totalTasks{ 0 };
	std::atomic<uint32_t> nextTaskIdx{ 0 };
	std::atomic<uint32_t> tasksCompleted{ 0 };

	std::mutex wakeMutex;
	std::condition_variable wakeCV;
	uint32_t workGeneration{ 0 };

	std::mutex doneMutex;
	std::condition_variable doneCV;

	void workerLoop(uint32_t threadId) {
		uint32_t localGeneration = 0;
		while (!terminatePool.load(std::memory_order_relaxed)) {
            {
                std::unique_lock<std::mutex> lock(wakeMutex);
                wakeCV.wait(lock, [&] { return terminatePool.load(std::memory_order_relaxed) || localGeneration != workGeneration; });

                if (terminatePool.load(std::memory_order_relaxed)) {
                    break;
                }

                localGeneration = workGeneration;
            }

			while (true) {
				uint32_t taskIdx = nextTaskIdx.fetch_add(1, std::memory_order_relaxed);

				if(taskIdx >= totalTasks) {
					break;
				}

				taskFunc(taskIdx);
				uint32_t completed = tasksCompleted.fetch_add(1, std::memory_order_release) + 1;

				if(completed == totalTasks) {
					std::lock_guard<std::mutex> lock(doneMutex);
					doneCV.notify_one();
				}
			}
		}
		
	}

public:
	JobSystem(uint32_t threadCount = std::thread::hardware_concurrency()) {
		numThreads = threadCount == 0 ? 8 : threadCount;

		for (uint32_t i = 0; i < numThreads; ++i) {
			workers.emplace_back(&JobSystem::workerLoop, this, i);
		}
	}

	~JobSystem() {
		terminatePool.store(true, std::memory_order_relaxed);
		{
			std::lock_guard<std::mutex> lock(wakeMutex);
			workGeneration++;
		}
		wakeCV.notify_all();

		for (auto& worker : workers) {
			if (worker.joinable()) {
				worker.join();
			}
		}
	}

	void Dispatch(uint32_t taskCount, std::function<void(uint32_t)> func) {
		if(taskCount == 0) return;
		
		
		taskFunc = func;
		totalTasks = taskCount;
		nextTaskIdx.store(0, std::memory_order_relaxed);
		tasksCompleted.store(0, std::memory_order_relaxed);
		{
			std::lock_guard<std::mutex> lock(wakeMutex);
			workGeneration++;
		}
		wakeCV.notify_all();

		{
			std::unique_lock<std::mutex> lock(doneMutex);
			doneCV.wait(lock, [&] { return tasksCompleted.load(std::memory_order_acquire) >= totalTasks; });
		}
	}

	uint32_t GetThreadCount() const {
		return numThreads;
	}
};