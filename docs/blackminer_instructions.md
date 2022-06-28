# Mining Nexus with Blackminer
## Software
* Mining with a Blackminer FPGA requires the proper bitstream for your Blackminer type

## Bitstream
* Download the proper bitstream from: https://hashaltcoin.gitbook.io/doc/ 
* If you can't find the Nexus bitstream for your device, it is likely you need to use a bitstream from a different device
* Join the HashAltcoin FPGA miner Discord if you have questions about what bitstream to use

## Pool
* `hashpool.nexus.io:50000`
* Nexus address must be a Tritium (non-legacy) address  
* Do not put a "." or user name after the address
* Below is an image of the Configuration>Pool/Miner page
![image](https://user-images.githubusercontent.com/108245170/175842579-c02e94ae-f2a2-4ff4-b294-d22d2aadc31b.png)

## Other
* If you are connected properly to the pool, your BlackMiner Miner Status page will show data
* Temporarily you must login to your Tritium Wallet every <24h to properly receive your mining rewards
* Why? Nexus implemented a feature that if you send your Nexus to the wrong address, it will automatically be returned to the sender in 24h if the wallet is not valid or synced and opened.  In the future, the default for this is changing to 7 days -or- will be configurable
* If you are staking you don't need to worry about this as your wallet is normally always open
* This is not a staking tutorial. However, as a reminder, in order to stake you need a certain amount of Nexus in your trust account. Otherwise, you won't get consistent stakes and your Nexus can be locked. This is changing with pooled staking in the future.

## Support
* [Nexus Miners](https://t.me/NexusMiners) on telegram for Nexus wallet and pool related support.
