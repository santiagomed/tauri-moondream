import SwiftRs
import Tauri
import UIKit
import WebKit
import Photos

class PingArgs: Decodable {
  let value: String?
}

class Image: Plugin {
  @objc public func ping(_ invoke: Invoke) throws {
    let args = try invoke.parseArgs(PingArgs.self)
    invoke.resolve(["value": args.value ?? ""])
  }
    
  @objc public override func checkPermissions(_ invoke: Invoke) {
    let status: PHAuthorizationStatus = PHPhotoLibrary.authorizationStatus()
    invoke.resolve(["granted": status == .authorized])
  }
    
  @objc public override func requestPermissions(_ invoke: Invoke) {
    let status = PHPhotoLibrary.authorizationStatus()
    if status == .notDetermined {
      PHPhotoLibrary.requestAuthorization { status in
        invoke.resolve(["granted": status == .authorized])
      }
    } else {
      invoke.resolve(["granted": status == .authorized])
    }
  }
}

@_cdecl("init_plugin_image")
func initPlugin() -> Plugin {
  return Image()
}
